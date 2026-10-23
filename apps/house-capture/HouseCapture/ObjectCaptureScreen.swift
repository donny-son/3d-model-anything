import ARKit
import SwiftUI

/// Full-screen object scan: tap the object to box it, then walk around it until the dots turn green.
struct ObjectCaptureScreen: View {
  @ObservedObject var controller: ObjectCaptureController
  let onClose: () -> Void
  @State private var confirmClose = false
  @State private var sharing: URL?
  @State private var zipping = false
  @State private var error: String?
  @Environment(\.scenePhase) private var scenePhase

  var body: some View {
    ZStack {
      ARViewContainer(view: controller.view).ignoresSafeArea()
        .onTapGesture { location in controller.place(at: location) }
      VStack(spacing: 10) {
        topBar
        Spacer()
        if let warning = controller.warning {
          Text(warning)
            .font(.footnote)
            .padding(10)
            .background(.orange.opacity(0.9), in: RoundedRectangle(cornerRadius: 10))
            .padding(.horizontal)
        }
        bottomPanel
      }
      .padding(.bottom, 8)
    }
    .onAppear { controller.begin() }
    .onChange(of: scenePhase) { _, phase in
      if phase != .active { controller.pause() }
    }
    .confirmationDialog("Discard this scan?", isPresented: $confirmClose, titleVisibility: .visible) {
      Button("Discard scan", role: .destructive) {
        controller.abandon()
        onClose()
      }
      Button("Keep scanning", role: .cancel) {}
    }
    .alert("Something went wrong", isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })) {
      Button("OK") { error = nil }
    } message: { Text(error ?? "") }
    .sheet(item: $sharing) { ShareSheet(items: [$0]) }
    .statusBarHidden()
  }

  // MARK: top

  private var topBar: some View {
    HStack(alignment: .top) {
      Button {
        switch controller.phase {
        case .done, .failed:
          controller.close()
          onClose()
        case .saving:
          break
        default:
          confirmClose = true
        }
      } label: {
        Image(systemName: "xmark")
          .font(.headline)
          .frame(width: 44, height: 44)
          .background(.ultraThinMaterial, in: Circle())
      }
      .accessibilityLabel("Close")
      Spacer()
      VStack(alignment: .trailing, spacing: 3) {
        Text(controller.name).font(.headline)
        Text("\(controller.depthFrameCount) depth frames · \(controller.photoCount) photos")
          .font(.caption.monospacedDigit())
        HStack(spacing: 8) {
          if let d = controller.distance, controller.phase == .scanning {
            Text("\(Int((d * 100).rounded())) cm away")
              .font(.caption2.weight(.semibold).monospacedDigit())
              .foregroundStyle(d < ObjectCaptureController.farthest && d > ObjectCaptureController.nearest ? Color.primary : .orange)
          }
          if controller.matFound {
            Label("Mat", systemImage: "checkerboard.rectangle").font(.caption2.weight(.semibold)).foregroundStyle(.green)
          }
        }
      }
      .padding(.horizontal, 12)
      .padding(.vertical, 8)
      .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
    .padding(.horizontal)
    .padding(.top, 8)
  }

  // MARK: bottom

  @ViewBuilder
  private var bottomPanel: some View {
    switch controller.phase {
    case .placing:
      panel {
        Label(controller.hint, systemImage: "hand.tap").font(.headline)
        Text(controller.matFound
             ? "Capture mat found: it sets the base plane and helps tracking."
             : "Tip: a printed capture mat under the object gives a flatter base and steadier tracking.")
          .font(.caption)
      }
    case .adjusting:
      panel {
        Text("Object box").font(.headline)
        Text(String(format: "W %.0f · D %.0f · H %.0f mm", controller.boxSize.x * 1000, controller.boxSize.z * 1000, controller.boxSize.y * 1000))
          .font(.subheadline.monospacedDigit())
        Text(controller.hint).font(.caption)
        LabeledContent("Footprint") {
          Slider(value: $controller.footprintScale, in: 0.5...2)
        }.font(.caption)
        LabeledContent("Height") {
          Slider(value: $controller.heightScale, in: 0.5...2)
        }.font(.caption)
        Button {
          controller.startScanning()
        } label: { Text("Start scanning").frame(maxWidth: .infinity) }
          .buttonStyle(.borderedProminent)
      }
    case .scanning:
      panel {
        HStack(spacing: 12) {
          CoverageRing(percent: Int((controller.coverage * 100).rounded())).frame(width: 48, height: 48)
            .accessibilityLabel("View coverage \(Int((controller.coverage * 100).rounded())) percent")
          Text(controller.hint).font(.subheadline)
        }
        HStack {
          Button(controller.paused ? "Resume" : "Pause") {
            if controller.paused { controller.resume() } else { controller.pause() }
          }
          .buttonStyle(.bordered)
          Spacer()
          Button("Finish") { Task { await controller.finish() } }
            .buttonStyle(.borderedProminent)
            .disabled(controller.depthFrameCount < 10)
        }
      }
    case .saving:
      panel { ProgressView("Saving…") }
    case .done:
      panel {
        Label("Saved \(controller.name)", systemImage: "checkmark.circle.fill").font(.headline)
        Text("\(controller.depthFrameCount) depth frames, \(controller.photoCount) photos, \(Int((controller.coverage * 100).rounded()))% of view directions.")
          .font(.subheadline)
        Text(verbatim: "Share the scan to your computer, unzip it, and run:\nobject_to_stl.py <scan folder>\n(scan-to-stl skill in this repository) to get the STL.")
          .font(.caption.monospaced())
        Button {
          share()
        } label: { Label(zipping ? "Zipping…" : "Share scan (.zip)", systemImage: "square.and.arrow.up").frame(maxWidth: .infinity) }
          .buttonStyle(.borderedProminent)
          .disabled(zipping)
        Button("Done") {
          controller.close()
          onClose()
        }
        .buttonStyle(.bordered)
      }
    case .failed(let message):
      panel {
        Text(message).font(.subheadline)
        if controller.hasRecording {
          Text("What was recorded is kept in the Objects list, where you can share or delete it.").font(.caption)
        }
        Button("Close") {
          if controller.hasRecording { controller.close() } else { controller.abandon() }
          onClose()
        }
        .buttonStyle(.bordered)
      }
    }
  }

  private func panel<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
    VStack(alignment: .leading, spacing: 10, content: content)
      .padding(14)
      .frame(maxWidth: 500, alignment: .leading)
      .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
      .padding(.horizontal)
  }

  private func share() {
    zipping = true
    let scan = ObjectScan(url: controller.scan.url, manifest: ObjectScan.loadManifest(controller.scan.url))
    Task.detached {
      let result = Result { try scan.zipped() }
      await MainActor.run {
        zipping = false
        switch result {
        case .success(let url): sharing = url
        case .failure(let e): error = e.localizedDescription
        }
      }
    }
  }
}

/// Saved object scans: share them to a computer for the STL pipeline, or start a new one
struct ObjectLibraryScreen: View {
  @State private var scans: [ObjectScan] = []
  @State private var capture: ObjectCaptureController?
  @State private var naming = false
  @State private var newName = ""
  @State private var error: String?
  @State private var sharing: URL?
  @State private var busy: String?
  @State private var deleting: ObjectScan?

  var body: some View {
    ScrollView {
      LazyVStack(alignment: .leading, spacing: 0) {
        VStack(alignment: .leading, spacing: 10) {
          FieldworkEyebrow(text: "Small objects")
          Text("Scan to print").font(.largeTitle.weight(.bold)).tracking(-1)
          Text("LiDAR depth, the exact position of every frame and the phone's motion, fused into a watertight STL on your computer.")
            .foregroundStyle(Fieldwork.secondary)
        }.padding(.top, 12).padding(.bottom, 28)
        FieldworkRule()
        tips.padding(.vertical, 20)
        FieldworkRule()
        if scans.isEmpty {
          VStack(alignment: .leading, spacing: 14) {
            Image(systemName: "cube.transparent").font(.system(size: 42, weight: .ultraLight)).foregroundStyle(Fieldwork.orange)
            Text("No object scans yet.").font(.title2.bold())
            Text("Scans stay on this iPhone until you share them.").foregroundStyle(Fieldwork.secondary)
          }.padding(.vertical, 32)
        }
        ForEach(scans) { scan in
          row(scan)
          FieldworkRule()
        }
        if !ObjectCaptureController.isSupported {
          Text("Object scanning needs an iPhone or iPad with LiDAR.")
            .font(.footnote).foregroundStyle(Fieldwork.secondary).padding(.vertical, 24)
        }
      }.padding(.horizontal, 24).padding(.bottom, 24)
    }
    .disabled(busy != nil)
    .overlay { if let busy { ProgressView(busy).padding(24).background(Fieldwork.paper, in: RoundedRectangle(cornerRadius: 12)) } }
    .background(Fieldwork.paper).foregroundStyle(Fieldwork.ink)
    .navigationTitle("").navigationBarTitleDisplayMode(.inline)
    .toolbar { ToolbarItem(placement: .principal) { FieldworkEyebrow(text: "Objects") } }
    .safeAreaInset(edge: .bottom) {
      Button {
        newName = "Object \(scans.count + 1)"
        naming = true
      } label: { Label("Scan an object", systemImage: "viewfinder") }
        .buttonStyle(FieldworkButtonStyle())
        .disabled(!ObjectCaptureController.isSupported || busy != nil)
        .padding(.horizontal, 24).padding(.top, 12).padding(.bottom, 10).background(Fieldwork.paper)
    }
    .onAppear(perform: reload)
    .alert("New object", isPresented: $naming) {
      TextField("Name", text: $newName)
      Button("Start scanning", action: start)
      Button("Cancel", role: .cancel) {}
    } message: {
      Text("Put it on a table, on the printed capture mat if you have one, with room to walk all the way around.")
    }
    .confirmationDialog("Delete \(deleting?.name ?? "scan")?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } }), titleVisibility: .visible) {
      Button("Delete scan", role: .destructive) {
        if let scan = deleting {
          do { try scan.delete() } catch { self.error = error.localizedDescription }
        }
        deleting = nil
        reload()
      }
    } message: { Text("This removes its depth frames, photos and motion data from this iPhone.") }
    .alert("Something went wrong", isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })) {
      Button("OK") { error = nil }
    } message: { Text(error ?? "") }
    .sheet(item: $sharing) { ShareSheet(items: [$0]) }
    .fullScreenCover(item: $capture, onDismiss: reload) { controller in
      ObjectCaptureScreen(controller: controller) { capture = nil }
    }
  }

  private var tips: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("For accurate prints").font(.headline)
      ForEach([
        "Matte, opaque objects from about 3 to 30 cm work best. Shiny, black or clear surfaces confuse the LiDAR.",
        "Set it on the printed capture mat (make_capture_mat.py) or a textured surface, away from other things.",
        "Keep 25–40 cm away and walk all the way around: low, level, then from above. Move slowly.",
        "Measure one dimension with calipers; the STL step can scale the model to match it exactly.",
      ], id: \.self) { tip in
        Label(tip, systemImage: "smallcircle.filled.circle").font(.footnote).foregroundStyle(Fieldwork.secondary)
      }
    }
  }

  private func row(_ scan: ObjectScan) -> some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(scan.name).font(.title3.bold())
      Text(summary(scan)).font(.footnote).foregroundStyle(Fieldwork.secondary)
      HStack {
        if scan.manifest != nil {
          Button { share(scan) } label: { Label("Share (.zip)", systemImage: "square.and.arrow.up") }
        }
        Spacer()
        Button(role: .destructive) { deleting = scan } label: { Label("Delete", systemImage: "trash") }
      }
      .font(.subheadline)
    }
    .padding(.vertical, 20)
  }

  private func summary(_ scan: ObjectScan) -> String {
    guard let m = scan.manifest else { return "Incomplete: not saved (\(scan.id)). It can't be converted; delete it to free space." }
    var parts = [m.createdAt.formatted(date: .abbreviated, time: .shortened), "\(m.depthFrameCount) depth frames", "\(m.photoCount) photos"]
    if let size = m.box?.size, size.count == 3 {
      parts.append(String(format: "box W %.0f · D %.0f · H %.0f mm", size[0] * 1000, size[2] * 1000, size[1] * 1000))
    }
    if let coverage = m.viewCoverage { parts.append("\(Int((coverage * 100).rounded()))% of views") }
    if m.mat != nil { parts.append("on mat") }
    return parts.joined(separator: " · ")
  }

  private func reload() {
    scans = ObjectScan.all()
  }

  private func start() {
    guard busy == nil else { return }
    let name = newName.trimmingCharacters(in: .whitespacesAndNewlines)
    busy = "Preparing…"
    Task {
      defer { busy = nil }
      do {
        try await CaptureReadiness.prepare()
        let controller = try ObjectCaptureController.start(name: name.isEmpty ? "Object" : name)
        try? await Task.sleep(for: .milliseconds(400)) // let the name alert finish dismissing
        capture = controller
      } catch { self.error = error.localizedDescription }
    }
  }

  private func share(_ scan: ObjectScan) {
    busy = "Zipping…"
    Task.detached {
      let result = Result { try scan.zipped() }
      await MainActor.run {
        busy = nil
        switch result {
        case .success(let url): sharing = url
        case .failure(let e): error = e.localizedDescription
        }
      }
    }
  }
}
