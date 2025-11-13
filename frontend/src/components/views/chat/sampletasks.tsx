import React, { useState, useEffect } from "react";

interface SampleTasksProps {
  onSelect: (task: string) => void;
}

const SAMPLE_TASKS = [
  "Create a sphere in ParaView using pvpython code",
  "Search for a bonsai image online, then load bonsai_256x256x256_uint8.raw and replicate it with volume visualization",
];

const SampleTasks: React.FC<SampleTasksProps> = ({ onSelect }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [windowWidth, setWindowWidth] = useState(0);

  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    handleResize(); // Initial width
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const isLargeScreen = windowWidth >= 1024; // lg breakpoint
  const tasksPerRow = windowWidth >= 640 ? 2 : 1; // 2 columns on sm, 1 on mobile
  const defaultVisibleTasks = tasksPerRow * 2;
  const maxVisibleTasks = isLargeScreen
    ? SAMPLE_TASKS.length
    : isExpanded
    ? SAMPLE_TASKS.length
    : defaultVisibleTasks;
  const visibleTasks = SAMPLE_TASKS.slice(0, maxVisibleTasks);
  const shouldShowToggle =
    !isLargeScreen && SAMPLE_TASKS.length > defaultVisibleTasks;

  return (
    <div className="mb-6">
      <div className="mt-4 mb-2 text-sm opacity-70 text-secondary">
        or try a sample task from below{" "}
      </div>
      <div className="flex flex-col gap-2 w-full">
        <div className="inline-flex flex-wrap justify-center gap-2 w-full">
          {visibleTasks.map((task, idx) => (
            <button
              key={idx}
              className="max-w-80 rounded px-4 py-2 text-left transition-colors text-primary hover:bg-secondary bg-tertiary"
              onClick={() => onSelect(task)}
              type="button"
            >
              {task}
            </button>
          ))}
        </div>
        {shouldShowToggle && (
          <button
            className="text-primary hover:text-secondary transition-colors text-sm font-medium mt-1"
            onClick={() => setIsExpanded(!isExpanded)}
            type="button"
          >
            {isExpanded ? "Show less..." : "Show more sample tasks..."}
          </button>
        )}
      </div>
    </div>
  );
};

export default SampleTasks;
