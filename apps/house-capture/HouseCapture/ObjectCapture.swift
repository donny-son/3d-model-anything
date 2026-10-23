import ARKit
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreMotion
import SceneKit
import SwiftUI

// MARK: - Bundle format (object-capture/1)

/// One small-object scan on disk (Documents/Objects/<yyyy-MM-dd HH.mm.ss>/), made to be fused
/// into a 3D-printable STL by `.agents/skills/scan-to-stl/scripts/object_to_stl.py`:
///
///     manifest.json          this struct: object box, support plane, capture mat, file index
///     depth/frames.json      LiDAR depth frames: pose + intrinsics (same records as house-capture/5, room 0)
///     depth/NNNNN.f16z       256×192 Float16 depth (m), raw DEFLATE; .conf UInt8 confidence 0–2; .jpg colour
///     photos/photos.json     posed photos (same records as house-capture/5, pass "object")
///     photos/NNNN.jpg        sensor orientation, EXIF 6
///     motion/imu.f64         100 Hz device motion: little-endian Float64 rows of `motion.fields`
///     features/points.json   ARKit's own 3D feature points inside the box, logged as you scan
///
/// Everything shares the scan's ARKit world: metres, +Y up (gravity), right-handed.
struct ObjectManifest: Codable {
  static let currentFormat = "object-capture/1"
  var format = ObjectManifest.currentFormat
  var name: String
  var createdAt: Date
  var endedAt: Date?
  var device: String
  var systemVersion: String
  var units = "meters"
  var coordinateSystem = "ARKit world space of this scan: meters, +Y up (gravity), right-handed. Box, support plane, mat, photo and depth poses all share it. Photo/depth transforms are camera-to-world (camera looks down -Z); intrinsics match the saved pixels (sensor orientation)."
  var box: ObjectBox?
  var supportPlane: SupportPlane?
  var mat: MatRecord?
  var depthFrames = "depth/frames.json"
  var depthFrameCount = 0
  var photos = "photos/photos.json"
  var photoCount = 0
  var motion: MotionRecord?
  var features: FeatureLog?
  /// fraction of the 36 view directions (12 around × 3 heights) with at least 3 depth frames
  var viewCoverage: Float?
  var events: [Diagnostics.Event]?
}

struct ObjectBox: Codable {
  /// object frame → world, column-major 4×4. Origin: centre of the box's bottom face, on the
  /// support plane; +Y = the plane's normal (up); X/Z along the box sides.
  var transform: [Float]
  /// width (x), height (y), depth (z) in metres; generous, the STL pipeline finds the object inside
  var size: [Float]
}

struct SupportPlane: Codable {
  var point: [Float]
  var normal: [Float]
  /// "mat" (printed capture mat), "plane-anchor" (ARKit plane), "depth" (table found in the
  /// LiDAR depth) or "raycast" (nothing found: a level plane through the tapped point)
  var source: String
  var classification: String?
}

/// The printed capture mat, tracked as an ARKit reference image of known width
struct MatRecord: Codable {
  var name: String
  /// width the app assumed for the printed pattern (m); check your print with a ruler
  var physicalWidth: Float
  /// last tracked pose: centre of the pattern, +Y out of the paper
  var transform: [Float]?
  /// median of ARKit's estimated / declared size while tracking (≈ 1 when ARKit's scale is right)
  var estimatedScaleFactor: Float?
  var samples: Int
  /// half the 10–90 % spread of those estimates
  var scaleSpread: Float?
}

struct MotionRecord: Codable {
  var file = "motion/imu.f64"
  var rateHz: Double
  var fields = MotionRecorder.fields
  var count: Int
  var note = "CMDeviceMotion. t: seconds since boot (the clock of ARFrame/photo/depth timestamps). q: attitude quaternion (xArbitraryZVertical reference). g: gravity (g units, device frame). r: rotation rate (rad/s, device frame). a: user acceleration (g units, device frame)."
}

struct FeatureLog: Codable {
  var file = "features/points.json"
  var frames: Int
  var note = "ARKit raw feature points inside the box (+2 cm), world coordinates, with their persistent identifiers."
}

struct FeatureFrame: Codable {
  var timestamp: TimeInterval
  var ids: [UInt64]
  var points: [[Float]]
}

/// A scan folder in Documents/Objects
struct ObjectScan: Identifiable {
  let url: URL
  var manifest: ObjectManifest?
  var id: String { url.lastPathComponent }
  var name: String { manifest?.name ?? id }

  static let subfolders = ["depth", "photos", "motion", "features"]

  static var root: URL {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Objects", isDirectory: true)
  }

  static func all() -> [ObjectScan] {
    let folders = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
    return folders
      .filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("manifest.json").path) }
      .map { ObjectScan(url: $0, manifest: loadManifest($0)) }
      .sorted { $0.id > $1.id }
  }

  static func create(name: String) throws -> ObjectScan {
    var id = HouseProject.folderName(for: Date())
    var n = 2
    while FileManager.default.fileExists(atPath: root.appendingPathComponent(id).path) {
      id = HouseProject.folderName(for: Date()) + " \(n)"
      n += 1
    }
    let url = root.appendingPathComponent(id, isDirectory: true)
    for dir in subfolders {
      try FileManager.default.createDirectory(at: url.appendingPathComponent(dir), withIntermediateDirectories: true)
    }
    return ObjectScan(url: url, manifest: nil)
  }

  static func loadManifest(_ url: URL) -> ObjectManifest? {
    guard let data = try? Data(contentsOf: url.appendingPathComponent("manifest.json")) else { return nil }
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    return try? decoder.decode(ObjectManifest.self, from: data)
  }

  func file(_ path: String) -> URL { url.appendingPathComponent(path) }

  func write<T: Encodable>(_ value: T, to path: String) throws {
    let e = JSONEncoder()
    e.outputFormatting = [.prettyPrinted, .sortedKeys]
    e.dateEncodingStrategy = .iso8601
    try e.encode(value).write(to: file(path), options: .atomic)
  }

  func delete() throws { try FileManager.default.removeItem(at: url) }

  func zipped() throws -> URL { try FolderZipper.zip(url, name: name) }
}

// MARK: - IMU

/// 100 Hz device motion: the latest rotation rate for live motion checks, and while recording,
/// rows of `fields` appended to motion/imu.f64. Timestamps share ARFrame's clock (seconds since
/// boot), so the STL pipeline can look up how fast the phone turned during each depth frame.
final class MotionRecorder: @unchecked Sendable {
  static let fields = ["t", "qx", "qy", "qz", "qw", "gx", "gy", "gz", "rx", "ry", "rz", "ax", "ay", "az"]
  static let rateHz = 100.0
  private let manager = CMMotionManager()
  private let queue: OperationQueue = {
    let q = OperationQueue()
    q.maxConcurrentOperationCount = 1
    q.qualityOfService = .userInitiated
    return q
  }()
  private let lock = NSLock()
  private var latestRate = 0.0
  private var rows = 0
  // queue only
  private var handle: FileHandle?
  private var pending = Data()

  var isAvailable: Bool { manager.isDeviceMotionAvailable }
  /// |rotation rate| of the latest sample, rad/s
  var rotationRate: Double { lock.withLock { latestRate } }
  var count: Int { lock.withLock { rows } }

  func start() {
    guard manager.isDeviceMotionAvailable, !manager.isDeviceMotionActive else { return }
    manager.deviceMotionUpdateInterval = 1 / Self.rateHz
    manager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: queue) { [weak self] motion, _ in
      guard let self, let motion else { return }
      self.append(motion)
    }
  }

  func record(to url: URL) {
    queue.addOperation {
      FileManager.default.createFile(atPath: url.path, contents: nil)
      self.handle = try? FileHandle(forWritingTo: url)
    }
  }

  func pauseRecording() {
    queue.addOperation {
      self.flush()
      try? self.handle?.close()
      self.handle = nil
    }
  }

  func resumeRecording(to url: URL) {
    queue.addOperation {
      guard self.handle == nil, let h = try? FileHandle(forWritingTo: url) else { return }
      _ = try? h.seekToEnd()
      self.handle = h
    }
  }

  /// Stop sampling and close the file (waits for the queue, a few milliseconds)
  func stop() {
    manager.stopDeviceMotionUpdates()
    queue.addOperation {
      self.flush()
      try? self.handle?.close()
      self.handle = nil
    }
    queue.waitUntilAllOperationsAreFinished()
  }

  private func append(_ m: CMDeviceMotion) {
    let q = m.attitude.quaternion, g = m.gravity, r = m.rotationRate, a = m.userAcceleration
    lock.withLock { latestRate = (r.x * r.x + r.y * r.y + r.z * r.z).squareRoot() }
    guard handle != nil else { return }
    let row: [Double] = [m.timestamp, q.x, q.y, q.z, q.w, g.x, g.y, g.z, r.x, r.y, r.z, a.x, a.y, a.z]
    row.withUnsafeBufferPointer { pending.append($0) } // little-endian on every iOS device
    lock.withLock { rows += 1 }
    if pending.count >= 64 * 1024 { flush() }
  }

  private func flush() {
    guard let handle, !pending.isEmpty else { return }
    try? handle.write(contentsOf: pending)
    pending.removeAll(keepingCapacity: true)
  }
}

// MARK: - Capture

/// Scans one small object for printing on a single ARSession:
///
/// - Find the table (an ARKit plane, or the printed capture mat), then tap the object: a box is
///   sized from the LiDAR depth around the tap and drawn over it.
/// - Walk around it: LiDAR depth frames (up to 10/s) with exact pose + intrinsics, sharp posed
///   photos for each new viewpoint, 100 Hz IMU and ARKit feature points inside the box. Frames
///   are only kept while tracking is normal, the object is in view, 12–80 cm away and the phone
///   is turning slowly enough not to smear the depth.
/// - Dots around the box turn green as each view direction (12 around × 3 heights) is covered.
@MainActor
final class ObjectCaptureController: NSObject, ObservableObject {
  enum Phase: Equatable {
    case placing, adjusting, scanning, saving, done
    case failed(String)
  }

  struct PlaneEstimate {
    var point: SIMD3<Float>
    var normal: SIMD3<Float>
    var source: String
    var classification: String?
  }

  @Published private(set) var phase: Phase = .placing
  @Published private(set) var hint = "Starting the camera…"
  @Published private(set) var distance: Float?
  @Published private(set) var coverage: Float = 0
  @Published private(set) var depthFrameCount = 0
  @Published private(set) var photoCount = 0
  @Published private(set) var surfaceFound = false
  @Published private(set) var matFound = false
  @Published private(set) var paused = false
  @Published private(set) var warning: String?
  /// box size as shown (metres): the estimate × the sliders
  @Published private(set) var boxSize = SIMD3<Float>(repeating: 0.1)
  @Published var footprintScale: Float = 1 { didSet { applyBox() } }
  @Published var heightScale: Float = 1 { didSet { applyBox() } }

  let scan: ObjectScan
  let name: String
  let arSession = ARSession()
  let view = ARSCNView(frame: .zero)
  private let overlay = ObjectOverlay()
  private let motion = MotionRecorder()
  private let cameraControl = CameraControl()
  private lazy var depthRecorder = DepthRecorder(bundle: ScanBundle(url: scan.url))
  private let ciContext = CIContext()
  private var displayLink: CADisplayLink?
  private var lastFrameTime: TimeInterval = 0
  private var lastPublish: TimeInterval = 0
  private var previousPose: (time: TimeInterval, pose: simd_float4x4)?
  private let startedAt = Date()
  private var events: [Diagnostics.Event] = []
  private var began = false
  private var closed = false

  // box
  private var plane: PlaneEstimate?
  private var boxFrame: simd_float4x4?
  private var baseSize = SIMD3<Float>(repeating: 0.1)

  // recording
  private var cellFrames = [Int](repeating: 0, count: 36)
  private var cellPhotos = [Int](repeating: 0, count: 36)
  private var photos: [PhotoRecord] = []
  private var photosInFlight = 0
  private var nextPhotoID = 0
  private var lastPhotoTime: TimeInterval = 0
  private var features: [FeatureFrame] = []
  private var lastFeatureTime: TimeInterval = 0
  private var matTransform: simd_float4x4?
  private var matScales: [Float] = []
  private var lastMatSample: TimeInterval = 0
  private var whiteBalanceAt: TimeInterval?

  static let matName = "capture-mat"
  /// printed width of CaptureMat.png's pattern; must match make_capture_mat.py
  static let matPhysicalWidth: Float = 0.18
  static let nearest: Float = 0.12
  static let farthest: Float = 0.8
  /// rad/s: faster than this the LiDAR depth smears (the STL pipeline uses the same limit)
  static let maxRotationRate = 1.2
  static let photoRotationRate = 0.35

  static var isSupported: Bool { ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) }

  init(scan: ObjectScan, name: String) {
    self.scan = scan
    self.name = name
    super.init()
    view.session = arSession
    view.scene = SCNScene()
    view.automaticallyUpdatesLighting = false
    view.rendersCameraGrain = false
    view.rendersMotionBlur = false
    view.antialiasingMode = .multisampling4X
    view.scene.rootNode.addChildNode(overlay.root)
  }

  static func start(name: String) throws -> ObjectCaptureController {
    try CaptureReadiness.requireStorage(at: FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0])
    let scan = try ObjectScan.create(name: name)
    return ObjectCaptureController(scan: scan, name: name)
  }

  /// Call when the capture screen appears
  func begin() {
    guard !began else { return }
    began = true
    let config = ARWorldTrackingConfiguration()
    config.planeDetection = [.horizontal]
    if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) { config.frameSemantics.insert(.sceneDepth) }
    let mat = Self.matReferenceImage()
    if let mat {
      config.detectionImages = [mat]
      config.maximumNumberOfTrackedImages = 1
      config.automaticImageScaleEstimationEnabled = true
    }
    arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
    motion.start()
    logEvent("started: depth=\(config.frameSemantics.contains(.sceneDepth)) mat=\(mat != nil) imu=\(motion.isAvailable)")
    let link = CADisplayLink(target: self, selector: #selector(tick))
    link.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
    link.add(to: .main, forMode: .common)
    displayLink = link
  }

  static func matReferenceImage() -> ARReferenceImage? {
    guard let image = UIImage(named: "CaptureMat")?.cgImage else { return nil }
    let reference = ARReferenceImage(image, orientation: .up, physicalWidth: CGFloat(matPhysicalWidth))
    reference.name = matName
    return reference
  }

  // MARK: placing the box

  /// Tap on the object: raycast to it, find the surface it stands on, size a box from the depth
  func place(at point: CGPoint) {
    guard phase == .placing || phase == .adjusting, let frame = arSession.currentFrame else { return }
    guard case .normal = frame.camera.trackingState else {
      warning = "Wait for tracking, then tap the object again."
      return
    }
    guard let query = view.raycastQuery(from: point, allowing: .estimatedPlane, alignment: .any),
          let hit = arSession.raycast(query).first else {
      warning = "Couldn't measure that spot. Tap the middle of the object."
      return
    }
    warning = nil
    let p = simd_make_float3(hit.worldTransform.columns.3)
    let support = supportPlane(below: p, frame: frame)
    let n = support.normal
    // box axes: the mat's own, else facing the camera
    var x: SIMD3<Float>
    if support.source == "mat", let m = matTransform {
      x = simd_make_float3(m.columns.0)
    } else {
      let back = simd_make_float3(frame.camera.transform.columns.2) // camera +Z points back at you
      let z = simd_normalize(back - simd_dot(back, n) * n)
      x = simd_cross(n, z)
    }
    x = simd_normalize(x - simd_dot(x, n) * n)
    let z = simd_cross(x, n)
    let onPlane = p - simd_dot(p - support.point, n) * n
    let footprint = Self.footprint(frame: frame, around: onPlane, plane: support, x: x, z: z)
    let origin = onPlane + x * footprint.center.x + z * footprint.center.y
    boxFrame = simd_float4x4(columns: (SIMD4(x, 0), SIMD4(n, 0), SIMD4(z, 0), SIMD4(origin, 1)))
    baseSize = footprint.size
    plane = support
    footprintScale = 1
    heightScale = 1
    phase = .adjusting
    applyBox()
    logEvent("box \(Int(baseSize.x * 1000))×\(Int(baseSize.y * 1000))×\(Int(baseSize.z * 1000)) mm on \(support.source)")
  }

  private func applyBox() {
    boxSize = baseSize * SIMD3(footprintScale, heightScale, footprintScale)
    overlay.update(box: boxFrame, size: boxSize, cells: cellFrames)
  }

  /// The surface under the tap: the tracked capture mat; else an ARKit horizontal plane up to
  /// 40 cm below the tap (clearly below it first, so an object's own flat top isn't taken for the
  /// table); else the table found in the depth; else a level plane through the tap.
  private func supportPlane(below p: SIMD3<Float>, frame: ARFrame) -> PlaneEstimate {
    if let m = matTransform {
      let point = simd_make_float3(m.columns.3), normal = simd_normalize(simd_make_float3(m.columns.1))
      let local = m.inverse * SIMD4(p, 1)
      if normal.y > 0.9, abs(local.x) < Self.matPhysicalWidth, abs(local.z) < Self.matPhysicalWidth {
        return PlaneEstimate(point: point, normal: normal, source: "mat", classification: nil)
      }
    }
    let planes = frame.anchors.compactMap { $0 as? ARPlaneAnchor }.filter { $0.alignment == .horizontal }.map { anchor in
      PlaneEstimate(point: simd_make_float3(anchor.transform * SIMD4(anchor.center, 1)),
                    normal: simd_normalize(simd_make_float3(anchor.transform.columns.1)),
                    source: "plane-anchor", classification: String(describing: anchor.classification))
    }
    for clearance: Float in [0.003, -0.005] {
      let under = planes.filter {
        let height = simd_dot(p - $0.point, $0.normal)
        return $0.normal.y > 0.9 && height > clearance && height < 0.4
      }
      if let best = under.max(by: { $0.point.y < $1.point.y }) { return best }
    }
    if let y = Self.depthTableHeight(frame: frame, around: p) {
      return PlaneEstimate(point: SIMD3(p.x, y, p.z), normal: SIMD3(0, 1, 0), source: "depth", classification: nil)
    }
    return PlaneEstimate(point: p, normal: SIMD3(0, 1, 0), source: "raycast", classification: nil)
  }

  /// Calls `body` with the world position of every `step`-th LiDAR pixel 8 cm–1.5 m away
  private static func forEachDepthPoint(_ frame: ARFrame, step: Int, minConfidence: UInt8, _ body: (SIMD3<Float>) -> Void) {
    guard let sceneDepth = frame.sceneDepth else { return }
    let depth = sceneDepth.depthMap
    CVPixelBufferLockBaseAddress(depth, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
    let conf = sceneDepth.confidenceMap
    if let conf { CVPixelBufferLockBaseAddress(conf, .readOnly) }
    defer { if let conf { CVPixelBufferUnlockBaseAddress(conf, .readOnly) } }
    guard let base = CVPixelBufferGetBaseAddress(depth) else { return }
    let w = CVPixelBufferGetWidth(depth), h = CVPixelBufferGetHeight(depth)
    let row = CVPixelBufferGetBytesPerRow(depth)
    let confBase = conf.flatMap { CVPixelBufferGetBaseAddress($0) }
    let confRow = conf.map { CVPixelBufferGetBytesPerRow($0) } ?? 0
    let scale = Float(w) / Float(frame.camera.imageResolution.width)
    let k = frame.camera.intrinsics
    let fx = k.columns.0.x * scale, fy = k.columns.1.y * scale, cx = k.columns.2.x * scale, cy = k.columns.2.y * scale
    let cam = frame.camera.transform
    for v in stride(from: 0, to: h, by: step) {
      let src = base.advanced(by: v * row).assumingMemoryBound(to: Float32.self)
      let c = confBase?.advanced(by: v * confRow).assumingMemoryBound(to: UInt8.self)
      for u in stride(from: 0, to: w, by: step) {
        let d = src[u]
        guard d > 0.08, d < 1.5, (c?[u] ?? 2) >= minConfidence else { continue }
        body(simd_make_float3(cam * SIMD4<Float>((Float(u) - cx) * d / fx, -(Float(v) - cy) * d / fy, -d, 1)))
      }
    }
  }

  /// Table height near `p` from the depth alone: the lowest well-populated 2 mm layer of LiDAR
  /// points within 25 cm of the tap and up to 40 cm below it
  private static func depthTableHeight(frame: ARFrame, around p: SIMD3<Float>) -> Float? {
    let layer: Float = 0.002
    var bins = [Int](repeating: 0, count: 200)
    forEachDepthPoint(frame, step: 3, minConfidence: 1) { q in
      let below = p.y + 0.003 - q.y, dx = q.x - p.x, dz = q.z - p.z
      guard below >= 0, below < 0.4, dx * dx + dz * dz < 0.0625 else { return }
      bins[min(bins.count - 1, Int(below / layer))] += 1
    }
    let threshold = max(40, Int(Float(bins.max() ?? 0) * 0.4))
    guard let lowest = bins.indices.last(where: { bins[$0] >= threshold }) else { return nil }
    return p.y + 0.003 - (Float(lowest) + 0.5) * layer
  }

  /// Footprint of whatever stands on the plane around `center`: LiDAR points 4 mm–40 cm above it,
  /// binned in 5 mm cells and grown from the tapped cell. Returns the footprint centre (x, z offset
  /// from `center`) and a box size with 1 cm of margin.
  private static func footprint(frame: ARFrame, around center: SIMD3<Float>, plane: PlaneEstimate,
                                x: SIMD3<Float>, z: SIMD3<Float>) -> (center: SIMD2<Float>, size: SIMD3<Float>) {
    let fallback = (center: SIMD2<Float>(0, 0), size: SIMD3<Float>(0.12, 0.12, 0.12))
    let cell: Float = 0.005
    struct Cell: Hashable { var a: Int; var b: Int }
    var counts: [Cell: Int] = [:]
    var tops: [Cell: Float] = [:]
    forEachDepthPoint(frame, step: 2, minConfidence: 2) { point in
      let height = simd_dot(point - plane.point, plane.normal)
      guard height > 0.004, height < 0.4 else { return }
      let offset = point - center
      let a = simd_dot(offset, x), b = simd_dot(offset, z)
      guard a * a + b * b < 0.25 * 0.25 else { return }
      let key = Cell(a: Int((a / cell).rounded(.down)), b: Int((b / cell).rounded(.down)))
      counts[key, default: 0] += 1
      tops[key] = max(tops[key] ?? 0, height)
    }
    let occupied = Set(counts.filter { $0.value >= 2 }.keys)
    // start at the tapped cell, or the nearest occupied one
    let start = occupied.min { ($0.a * $0.a + $0.b * $0.b) < ($1.a * $1.a + $1.b * $1.b) }
    guard let start, start.a * start.a + start.b * start.b <= 25 else { return fallback }
    var seen: Set<Cell> = [start]
    var queue = [start]
    while let next = queue.popLast() {
      for da in -1...1 {
        for db in -1...1 {
          let n = Cell(a: next.a + da, b: next.b + db)
          if occupied.contains(n), seen.insert(n).inserted { queue.append(n) }
        }
      }
    }
    let minA = Float(seen.map(\.a).min()!) * cell, maxA = Float(seen.map(\.a).max()! + 1) * cell
    let minB = Float(seen.map(\.b).min()!) * cell, maxB = Float(seen.map(\.b).max()! + 1) * cell
    let top = seen.compactMap { tops[$0] }.max() ?? 0.1
    let margin: Float = 0.01
    func clamp(_ v: Float) -> Float { min(0.5, max(0.03, v)) }
    return (SIMD2((minA + maxA) / 2, (minB + maxB) / 2),
            SIMD3(clamp(maxA - minA + 2 * margin), clamp(top + margin), clamp(maxB - minB + 2 * margin)))
  }

  // MARK: scanning

  func startScanning() {
    guard phase == .adjusting, boxFrame != nil else { return }
    do { try CaptureReadiness.requireStorage(at: scan.url) } catch {
      warning = error.localizedDescription
      return
    }
    motion.record(to: scan.file("motion/imu.f64"))
    phase = .scanning
    logEvent("scanning")
  }

  func pause() {
    guard phase == .scanning, !paused else { return }
    paused = true
    motion.pauseRecording()
    logEvent("paused")
  }

  func resume() {
    guard phase == .scanning, paused else { return }
    paused = false
    previousPose = nil
    motion.resumeRecording(to: scan.file("motion/imu.f64"))
    logEvent("resumed")
  }

  @objc private func tick() {
    guard !closed, let frame = arSession.currentFrame, frame.timestamp != lastFrameTime else { return }
    lastFrameTime = frame.timestamp
    let now = frame.timestamp
    trackAnchors(frame)
    let poseRate = Self.poseRate(frame, previous: previousPose)
    previousPose = (frame.timestamp, frame.camera.transform)
    let rate = motion.isAvailable ? motion.rotationRate : poseRate
    var normal = false
    if case .normal = frame.camera.trackingState { normal = true }
    if whiteBalanceAt == nil, normal { whiteBalanceAt = now + 3 }
    if let t = whiteBalanceAt, now > t, !cameraControl.whiteBalanceLocked {
      cameraControl.lockWhiteBalance()
      logEvent("white balance \(cameraControl.summary ?? "")")
    }
    switch phase {
    case .placing:
      setHint(!normal ? CaptureController.trackingText(frame.camera.trackingState)
              : surfaceFound || matFound ? "Tap the object you want to print."
              : "Point at the table the object stands on and move the phone slowly.")
    case .adjusting:
      setHint("The box should hold the whole object with a little room. Tap again to move it.")
    case .scanning:
      scanTick(frame, normal: normal, rate: rate)
    default:
      break
    }
  }

  private func setHint(_ text: String) {
    if hint != text { hint = text }
  }

  private func trackAnchors(_ frame: ARFrame) {
    if !surfaceFound, frame.anchors.contains(where: { ($0 as? ARPlaneAnchor)?.alignment == .horizontal }) {
      surfaceFound = true
    }
    guard let mat = frame.anchors.lazy.compactMap({ $0 as? ARImageAnchor }).first(where: { $0.referenceImage.name == Self.matName }) else { return }
    if !matFound {
      matFound = true
      logEvent("capture mat found")
    }
    guard mat.isTracked else { return }
    matTransform = mat.transform
    if frame.timestamp - lastMatSample > 0.5, matScales.count < 2000 {
      lastMatSample = frame.timestamp
      matScales.append(Float(mat.estimatedScaleFactor))
    }
  }

  private func scanTick(_ frame: ARFrame, normal: Bool, rate: Double) {
    guard let box = boxFrame else { return }
    let camPos = simd_make_float3(frame.camera.transform.columns.3)
    let center = simd_make_float3(box * SIMD4<Float>(0, boxSize.y / 2, 0, 1))
    let dist = simd_distance(camPos, center)
    if frame.timestamp - lastPublish > 0.15 {
      lastPublish = frame.timestamp
      distance = dist
      depthFrameCount = depthRecorder.records.count
    }
    guard !paused else { return setHint("Paused. Resume when you're back where you stopped.") }
    guard normal else { return setHint(CaptureController.trackingText(frame.camera.trackingState)) }
    guard let pixel = Self.project(center, frame: frame),
          pixel.x > 0.12 * frame.camera.imageResolution.width, pixel.x < 0.88 * frame.camera.imageResolution.width,
          pixel.y > 0.12 * frame.camera.imageResolution.height, pixel.y < 0.88 * frame.camera.imageResolution.height
    else { return setHint("Keep the object in the middle of the screen.") }
    guard dist < Self.farthest else { return setHint("Move closer: 25–40 cm from the object is best.") }
    guard dist > Self.nearest else { return setHint("Back up a little: the LiDAR needs about 15 cm.") }
    guard rate < Self.maxRotationRate else { return setHint("Slow down.") }
    setHint(coverage < 0.75 ? "Walk slowly all the way around: low, at eye level with it, then from above."
            : "Good coverage. Fill in any orange dots, then tap Finish.")
    let critical = [.serious, .critical].contains(ProcessInfo.processInfo.thermalState)
    let cell = Self.viewCell(camPos, box: box, center: center)
    let kept = depthRecorder.consider(frame, room: 0, camera: cameraControl.state(for: frame),
                                      interval: critical ? 0.3 : 0.1, minMove: 0.01, minTurn: 0.04)
    if kept {
      cellFrames[cell] += 1
      if cellFrames[cell] == 3 {
        coverage = Float(cellFrames.filter { $0 >= 3 }.count) / Float(cellFrames.count)
        overlay.update(box: boxFrame, size: boxSize, cells: cellFrames)
      }
      logFeatures(frame, box: box)
    }
    if photosInFlight == 0, cellPhotos[cell] < 2, frame.timestamp - lastPhotoTime > 0.6, rate < Self.photoRotationRate {
      takePhoto(frame, cell: cell, rate: rate)
    }
  }

  /// Camera turn rate from consecutive poses (when the IMU isn't available)
  private static func poseRate(_ frame: ARFrame, previous: (time: TimeInterval, pose: simd_float4x4)?) -> Double {
    guard let previous, frame.timestamp > previous.time else { return 0 }
    let delta = simd_quatf(previous.pose).inverse * simd_quatf(frame.camera.transform)
    return Double(2 * acos(min(1, abs(delta.real)))) / (frame.timestamp - previous.time)
  }

  private static func project(_ p: SIMD3<Float>, frame: ARFrame) -> CGPoint? {
    let c = frame.camera.transform.inverse * SIMD4<Float>(p, 1)
    guard c.z < -0.01 else { return nil }
    let k = frame.camera.intrinsics
    return CGPoint(x: CGFloat(k.columns.0.x * c.x / -c.z + k.columns.2.x),
                   y: CGFloat(k.columns.1.y * -c.y / -c.z + k.columns.2.y))
  }

  /// 12 directions around × 3 heights (below 30°, 30–60°, above 60°) seen from the box centre
  private static func viewCell(_ camera: SIMD3<Float>, box: simd_float4x4, center: SIMD3<Float>) -> Int {
    let local = simd_make_float3(box.inverse * SIMD4(camera, 1)) - simd_make_float3(box.inverse * SIMD4(center, 1))
    var azimuth = atan2(local.z, local.x) * 180 / .pi
    if azimuth < 0 { azimuth += 360 }
    let elevation = atan2(local.y, (local.x * local.x + local.z * local.z).squareRoot()) * 180 / .pi
    let band = elevation < 30 ? 0 : elevation < 60 ? 1 : 2
    return band * 12 + min(11, Int(azimuth / 30))
  }

  private func logFeatures(_ frame: ARFrame, box: simd_float4x4) {
    guard frame.timestamp - lastFeatureTime > 0.25, features.count < 3000, let cloud = frame.rawFeaturePoints else { return }
    lastFeatureTime = frame.timestamp
    let inverse = box.inverse, margin: Float = 0.02
    var ids: [UInt64] = [], points: [[Float]] = []
    for (p, id) in zip(cloud.points, cloud.identifiers) {
      let l = inverse * SIMD4<Float>(p, 1)
      guard abs(l.x) <= boxSize.x / 2 + margin, abs(l.z) <= boxSize.z / 2 + margin,
            l.y >= -margin, l.y <= boxSize.y + margin else { continue }
      ids.append(id)
      points.append([p.x, p.y, p.z])
    }
    if !ids.isEmpty { features.append(FeatureFrame(timestamp: frame.timestamp, ids: ids, points: points)) }
  }

  private func takePhoto(_ frame: ARFrame, cell: Int, rate: Double) {
    guard let buffer = CaptureController.copy(frame.capturedImage) else { return }
    photosInFlight += 1
    lastPhotoTime = frame.timestamp
    cellPhotos[cell] += 1
    let k = frame.camera.intrinsics
    let exposure = frame.camera.exposureDuration > 0 ? frame.camera.exposureDuration : 1.0 / 60
    let record = PhotoRecord(
      file: String(format: "photos/%04d.jpg", nextPhotoID), room: 0, timestamp: frame.timestamp,
      width: Int(frame.camera.imageResolution.width), height: Int(frame.camera.imageResolution.height),
      transform: frame.camera.transform.columnMajor, intrinsics: k.columnMajor, tracking: "normal", manual: false,
      exposure: exposure, blurPixels: Float(rate * exposure) * k.columns.0.x, pass: "object", highRes: false,
      camera: cameraControl.state(for: frame))
    nextPhotoID += 1
    let url = scan.file(record.file), context = ciContext
    let image = CIImage(cvPixelBuffer: buffer)
    Task.detached(priority: .utility) {
      let ok = CaptureController.writeJPEG(image, to: url, context: context)
      await MainActor.run {
        self.photosInFlight -= 1
        guard !self.closed else { return }
        if ok {
          self.photos.append(record)
          self.photoCount = self.photos.count
        } else {
          self.warning = "A photo could not be saved. Check available storage."
        }
      }
    }
  }

  // MARK: finishing

  func finish() async {
    guard phase == .scanning else { return }
    phase = .saving
    stopTicking()
    motion.stop()
    let deadline = Date().addingTimeInterval(5)
    while (depthRecorder.isWriting || photosInFlight > 0) && Date() < deadline {
      try? await Task.sleep(for: .milliseconds(50))
    }
    arSession.pause()
    do {
      try depthRecorder.save()
      try scan.write(photos, to: "photos/photos.json")
      try scan.write(features, to: "features/points.json")
      try scan.write(manifest(), to: "manifest.json")
      depthFrameCount = depthRecorder.records.count
      phase = .done
    } catch {
      phase = .failed("Couldn't save the scan: \(error.localizedDescription)")
    }
  }

  private func manifest() -> ObjectManifest {
    var m = ObjectManifest(name: name, createdAt: startedAt, device: UIDevice.modelIdentifier, systemVersion: UIDevice.current.systemVersion)
    m.endedAt = Date()
    if let box = boxFrame {
      m.box = ObjectBox(transform: box.columnMajor, size: [boxSize.x, boxSize.y, boxSize.z])
    }
    if let plane {
      m.supportPlane = SupportPlane(point: [plane.point.x, plane.point.y, plane.point.z],
                                    normal: [plane.normal.x, plane.normal.y, plane.normal.z],
                                    source: plane.source, classification: plane.classification)
    }
    if matFound {
      let sorted = matScales.sorted()
      func quantile(_ q: Float) -> Float? { sorted.isEmpty ? nil : sorted[min(sorted.count - 1, Int(Float(sorted.count - 1) * q))] }
      var spread: Float?
      if let hi = quantile(0.9), let lo = quantile(0.1) { spread = (hi - lo) / 2 }
      m.mat = MatRecord(name: Self.matName, physicalWidth: Self.matPhysicalWidth, transform: matTransform?.columnMajor,
                        estimatedScaleFactor: quantile(0.5), samples: sorted.count, scaleSpread: spread)
    }
    m.depthFrameCount = depthRecorder.records.count
    m.photoCount = photos.count
    if motion.count > 0 { m.motion = MotionRecord(rateHz: MotionRecorder.rateHz, count: motion.count) }
    m.features = FeatureLog(frames: features.count)
    m.viewCoverage = coverage
    m.events = events
    return m
  }

  /// Leave without saving: the scan folder is deleted
  func abandon() {
    closed = true
    stopTicking()
    motion.stop()
    arSession.pause()
    try? scan.delete()
  }

  /// Leave after saving (or after a failed save, keeping what was written)
  func close() {
    closed = true
    stopTicking()
    motion.stop()
    arSession.pause()
  }

  private func stopTicking() {
    displayLink?.invalidate()
    displayLink = nil
  }

  private func logEvent(_ what: String) {
    events.append(Diagnostics.Event(t: (Date().timeIntervalSince(startedAt) * 10).rounded() / 10, what: what))
  }
}

extension ObjectCaptureController: Identifiable {}

// MARK: - AR overlay

/// The object box as a wireframe, plus one dot per view direction around it
/// (orange: needs depth frames from there, green: covered).
@MainActor
final class ObjectOverlay {
  let root = SCNNode()
  private let wire = SCNNode()
  private let dots = SCNNode()
  private var dotNodes: [SCNNode] = []
  private let todo: SCNMaterial = ObjectOverlay.material(UIColor(red: 1, green: 0.45, blue: 0.15, alpha: 1))
  private let done: SCNMaterial = ObjectOverlay.material(UIColor(red: 0.25, green: 0.9, blue: 0.45, alpha: 1))

  init() {
    root.isHidden = true
    root.addChildNode(wire)
    root.addChildNode(dots)
    // unit box, bottom face centred on the origin
    var corners: [SCNVector3] = []
    for y in [Float(0), 1] {
      for (x, z) in [(Float(-0.5), Float(-0.5)), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] { corners.append(SCNVector3(x, y, z)) }
    }
    let edges: [Int32] = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]
    let geometry = SCNGeometry(sources: [SCNGeometrySource(vertices: corners)],
                               elements: [SCNGeometryElement(indices: edges, primitiveType: .line)])
    geometry.materials = [Self.material(.white)]
    wire.geometry = geometry
    let sphere = SCNSphere(radius: 0.004)
    for _ in 0..<36 {
      let node = SCNNode(geometry: sphere.copy() as? SCNGeometry)
      dots.addChildNode(node)
      dotNodes.append(node)
    }
  }

  nonisolated private static func material(_ color: UIColor) -> SCNMaterial {
    let m = SCNMaterial()
    m.lightingModel = .constant
    m.diffuse.contents = color
    m.readsFromDepthBuffer = false
    return m
  }

  func update(box: simd_float4x4?, size: SIMD3<Float>, cells: [Int]) {
    guard let box else {
      root.isHidden = true
      return
    }
    root.isHidden = false
    root.simdTransform = box
    wire.simdScale = size
    let radius = max(size.x, size.z) / 2 + 0.05
    for (i, node) in dotNodes.enumerated() {
      let band = i / 12, around = i % 12
      let azimuth = (Float(around) * 30 + 15) * .pi / 180, elevation = (Float(band) * 30 + 15) * .pi / 180
      node.simdPosition = SIMD3(radius * cos(elevation) * cos(azimuth), size.y / 2 + radius * sin(elevation),
                                radius * cos(elevation) * sin(azimuth))
      node.geometry?.materials = [i < cells.count && cells[i] >= 3 ? done : todo]
    }
  }
}
