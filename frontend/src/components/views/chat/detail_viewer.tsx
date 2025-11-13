import React, { useState, useRef, lazy, Suspense } from "react";
import {
  Maximize2,
  MousePointerClick,
  X,
} from "lucide-react";
import BrowserIframe from "./DetailViewer/browser_iframe";
import BrowserModal from "./DetailViewer/browser_modal";
import FullscreenOverlay from "./DetailViewer/fullscreen_overlay"; // Import our new component
import { IPlan } from "../../types/plan";
import { useSettingsStore } from "../../store";
import { RcFile } from "antd/es/upload";
// Define VNC component props type
interface VncScreenProps {
  url: string;
  scaleViewport?: boolean;
  background?: string;
  style?: React.CSSProperties;
  ref?: React.Ref<any>;
}
// Lazy load the VNC component
const VncScreen = lazy<React.ComponentType<VncScreenProps>>(() =>
  // @ts-ignore
  import("react-vnc").then((module) => ({ default: module.VncScreen }))
);

interface DetailViewerProps {
  images: string[];
  imageTitles: string[];
  onMinimize: () => void;
  onToggleExpand: () => void;
  isExpanded: boolean;
  currentIndex: number;
  onIndexChange: (index: number) => void;
  novncPort?: string;
  paraviewPort?: string;
  browserPort?: string;
  onPause?: () => void;
  runStatus?: string;
  activeTab?: TabType;
  onTabChange?: (tab: TabType) => void;
  detailViewerContainerId?: string;
  onInputResponse?: (
    response: string,
    files: RcFile[],
    accepted?: boolean,
    plan?: IPlan
  ) => void;
}

type TabType = "screenshots" | "paraview" | "browser";

const DetailViewer: React.FC<DetailViewerProps> = ({
  images,
  imageTitles,
  onMinimize,
  currentIndex,
  onIndexChange,
  novncPort,
  paraviewPort,
  browserPort,
  onPause,
  runStatus,
  activeTab: controlledActiveTab,
  onTabChange,
  detailViewerContainerId,
  onInputResponse,
}) => {
  const [internalActiveTab, setInternalActiveTab] = useState<TabType>("paraview");
  const activeTab = controlledActiveTab ?? internalActiveTab;
  const [viewMode, setViewMode] = useState<"iframe" | "novnc">("iframe");
  const vncRef = useRef();

  const [isModalOpen, setIsModalOpen] = useState(false);

  // Update active tab when ports become available
  React.useEffect(() => {
    // Only auto-switch if user hasn't manually changed tabs
    if (!controlledActiveTab) {
      if (paraviewPort && internalActiveTab !== "paraview") {
        setInternalActiveTab("paraview");
      } else if (browserPort && !paraviewPort && internalActiveTab !== "browser") {
        setInternalActiveTab("browser");
      }
    }
  }, [paraviewPort, browserPort, controlledActiveTab, internalActiveTab]);

  // Add state for fullscreen control mode
  const [isControlMode, setIsControlMode] = useState(false);
  const browserIframeId = "browser-iframe-container";

  // State for tracking if control was handed back from modal
  const [showControlHandoverForm, setShowControlHandoverForm] = useState(false);

  const config = useSettingsStore((state) => state.config);

  // Handle take control action
  const handleTakeControl = () => {
    setIsControlMode(true);
  };

  // Exit control mode
  const exitControlMode = () => {
    setIsControlMode(false);
  };

  // Modal control handlers
  const handleModalControlHandover = () => {
    // Show the feedback form overlay in DetailViewer
    setIsControlMode(true);
    setShowControlHandoverForm(true);
  };

  const handleTabChange = (tab: TabType) => {
    if (onTabChange) {
      onTabChange(tab);
    } else {
      setInternalActiveTab(tab);
    }
  };

  const handleMaximizeClick = () => {
    setIsModalOpen(true);
  };

  const renderVncView = (port: string, serviceName: string) => {
    // Use server_url from config if set, otherwise default to localhost
    const serverHost = config.server_url || "localhost";

    return (
      <div className="flex-1 w-full h-full flex flex-col">
        {viewMode === "iframe" ? (
          <BrowserIframe
            novncPort={port}
            style={{
              height: "100%",
              flex: "1 1 auto",
            }}
            className="w-full flex-1"
            showDimensions={true}
            onPause={onPause}
            runStatus={runStatus}
            quality={7}
            viewOnly={false}
            scaling="local"
            showTakeControlOverlay={!isControlMode}
            onTakeControl={handleTakeControl}
            isControlMode={isControlMode}
            serverUrl={serverHost}
          />
        ) : (
          <div
            className="relative w-full h-full flex flex-col"
            onMouseEnter={() => {}}
            onMouseLeave={() => {}}
          >
            <Suspense fallback={<div>Loading VNC viewer...</div>}>
              <VncScreen
                key={`vnc-${port}`}
                url={`ws://${serverHost}:${port}`}
                scaleViewport
                background="#000000"
                style={{
                  width: "100%",
                  height: "100%",
                  flex: "1 1 auto",
                  alignSelf: "flex-start",
                  display: "flex",
                  flexDirection: "column",
                }}
                ref={vncRef}
              />
            </Suspense>
          </div>
        )}
      </div>
    );
  };

  const renderParaViewTab = React.useMemo(() => {
    if (!paraviewPort) {
      return (
        <div className="flex items-center justify-center w-full h-full text-secondary">
          <div className="text-center">
            <div className="text-lg mb-2">Initializing ParaView...</div>
            <div className="text-sm">Please wait while the Docker container starts</div>
          </div>
        </div>
      );
    }

    return renderVncView(paraviewPort, "ParaView");
  }, [paraviewPort, viewMode, runStatus, onPause, isControlMode, config.server_url]);

  const renderBrowserTab = React.useMemo(() => {
    if (!browserPort) {
      return (
        <div className="flex-1 w-full h-full min-h-0 flex items-center justify-center">
          <p>Waiting for browser session to start...</p>
        </div>
      );
    }

    return renderVncView(browserPort, "Browser");
  }, [browserPort, viewMode, runStatus, onPause, isControlMode, config.server_url]);

  return (
    <>
      <div
        className="bg-tertiary rounded-lg shadow-lg p-4 h-full flex flex-col relative overflow-hidden"
        id={detailViewerContainerId}
      >
        {/* Tabs and Controls */}
        <div className="flex justify-between items-center mb-4 border-b flex-shrink-0">
          <div className="flex">
            {paraviewPort && (
              <button
                className={`px-6 py-2 font-medium ${
                  activeTab === "paraview"
                    ? "text-primary border-b-2 border-primary"
                    : "text-secondary hover:text-primary"
                }`}
                onClick={() => handleTabChange("paraview")}
              >
                ParaView
              </button>
            )}
            {browserPort && (
              <button
                className={`px-6 py-2 font-medium ${
                  activeTab === "browser"
                    ? "text-primary border-b-2 border-primary"
                    : "text-secondary hover:text-primary"
                }`}
                onClick={() => handleTabChange("browser")}
              >
                Browser
              </button>
            )}
            {!paraviewPort && !browserPort && (
              <div className="px-6 py-2 font-medium text-primary">
                Live View
              </div>
            )}
          </div>

          <div className="flex gap-2">
            {isControlMode && (
              <div className="flex items-center gap-2 px-2 rounded-2xl bg-magenta-800 text-white">
                <MousePointerClick size={16} />
                <span>You have control</span>
              </div>
            )}
            <button
              onClick={handleMaximizeClick}
              className="p-1 hover:bg-gray-100 rounded-full transition-colors"
              title="Open in full screen"
            >
              <Maximize2 size={20} />
            </button>
            {!isControlMode && (
              <button
                onClick={onMinimize}
                className="p-1 hover:bg-gray-100 rounded-full transition-colors"
              >
                <X size={20} />
              </button>
            )}
          </div>
        </div>

        <div className="flex-1 flex flex-col min-h-0">
          {activeTab === "paraview" && renderParaViewTab}
          {activeTab === "browser" && renderBrowserTab}
        </div>
      </div>

      <BrowserModal
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false);
        }}
        novncPort={activeTab === "paraview" ? paraviewPort : browserPort}
        title={activeTab === "paraview" ? "ParaView" : "Browser View"}
        onPause={onPause}
        runStatus={runStatus}
        onControlHandover={handleModalControlHandover}
        isControlMode={isControlMode}
        onTakeControl={handleTakeControl}
      />

      {/* Fullscreen Control Mode Overlay */}
      <FullscreenOverlay
        isVisible={isControlMode}
        onClose={() => {
          exitControlMode();
          setShowControlHandoverForm(false);
        }}
        targetElementId={detailViewerContainerId}
        zIndex={50}
        onInputResponse={onInputResponse}
        runStatus={runStatus}
      />
    </>
  );
};

export default DetailViewer;
