import React from "react";
import { PanelLeftClose, PanelLeftOpen, Plus, FileText } from "lucide-react";
import { Tooltip } from "antd";
import { appContext } from "../hooks/provider";
import { useConfigStore } from "../hooks/store";
import { Settings } from "lucide-react";
import SettingsModal from "./settings/SettingsModal";
import logo from "../assets/new_logo.png";
import { Button } from "./common/Button";

type ContentHeaderProps = {
  onMobileMenuToggle: () => void;
  isMobileMenuOpen: boolean;
  isSidebarOpen: boolean;
  onToggleSidebar: () => void;
  onNewSession: () => void;
  onNewPlan?: () => void;
};

const ContentHeader = ({
  isSidebarOpen,
  onToggleSidebar,
  onNewSession,
  onNewPlan,
}: ContentHeaderProps) => {
  const { user } = React.useContext(appContext);
  useConfigStore();
  const [isSettingsOpen, setIsSettingsOpen] = React.useState(false);

  return (
    <div className="sticky top-0 bg-primary">
      <div className="flex h-16 items-center justify-between">
        {/* Left side: Text and Sidebar Controls */}
        <div className="flex items-center">
          {/* Sidebar toggle removed - single session mode */}

          {/* New Session Button */}
          <div className="w-[40px]">
            <Tooltip title="Create new session">
              <Button
                variant="tertiary"
                size="sm"
                icon={<Plus className="w-6 h-6" />}
                onClick={onNewSession}
                className="transition-colors hover:text-accent"
              />
            </Tooltip>
          </div>

          {/* New Plan Button */}
          {onNewPlan && (
            <div className="w-[40px]">
              <Tooltip title="Create new plan">
                <Button
                  variant="tertiary"
                  size="sm"
                  icon={<FileText className="w-6 h-6" />}
                  onClick={onNewPlan}
                  className="transition-colors hover:text-accent"
                />
              </Tooltip>
            </div>
          )}

          <div className="flex items-center space-x-2">
            <img src={logo} alt="HILSVA Logo" className="h-10 w-10" />
            <div className="text-primary text-2xl font-bold">HILSVA</div>
          </div>
        </div>

        {/* Settings */}
        <div className="flex items-center">
          {/* Settings Button */}
          <div className="text-primary">
            <Tooltip title="Settings">
              <Button
                variant="tertiary"
                size="sm"
                icon={<Settings className="h-8 w-8" />}
                onClick={() => setIsSettingsOpen(true)}
                className="!px-0 transition-colors hover:text-accent"
                aria-label="Settings"
              />
            </Tooltip>
          </div>
        </div>
      </div>

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />
    </div>
  );
};

export default ContentHeader;
