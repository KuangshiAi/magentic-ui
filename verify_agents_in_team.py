#!/usr/bin/env python
"""Verify both ParaView agents are in the team."""

import asyncio
import yaml
from pathlib import Path

async def main():
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    from magentic_ui.magentic_ui_config import MagenticUIConfig
    from magentic_ui.task_team import get_task_team
    from magentic_ui.types import RunPaths

    # Create config
    magentic_config = MagenticUIConfig(**config_data)

    print("=" * 80)
    print("CONFIG LOADED")
    print("=" * 80)
    print(f"ParaView agent config: {magentic_config.paraview_agent_config is not None}")
    print(f"PVPython coder agent config: {magentic_config.pvpython_coder_agent_config is not None}")

    # Create paths
    from tempfile import mkdtemp
    temp_dir = Path(mkdtemp())
    paths = RunPaths(
        internal_root_dir=temp_dir,
        external_root_dir=temp_dir,
        internal_run_dir=temp_dir,
        external_run_dir=temp_dir,
        run_suffix="test_verify",
    )

    print("\n" + "=" * 80)
    print("CREATING TEAM")
    print("=" * 80)

    try:
        team = await get_task_team(
            magentic_ui_config=magentic_config,
            paths=paths,
        )
        print("✓ Team created successfully!")

        # Access participants
        if hasattr(team, '_participants'):
            participants = team._participants
            print(f"\nNumber of participants: {len(participants)}")
            print("\nParticipants:")
            for i, p in enumerate(participants):
                name = p.name if hasattr(p, 'name') else str(type(p).__name__)
                print(f"  {i+1}. {name}")

            # Check for our agents
            agent_names = [p.name.lower() for p in participants if hasattr(p, 'name')]
            paraview_found = any('paraview' in name for name in agent_names)
            pvpython_found = any('pvpython' in name for name in agent_names)

            print(f"\n{'✓' if paraview_found else '✗'} ParaView MCP agent (paraview_mcp) in team")
            print(f"{'✓' if pvpython_found else '✗'} PVPython Coder agent (pvpython_coder) in team")

            if not pvpython_found:
                print("\n⚠️  WARNING: PVPython Coder agent NOT found in team!")
                print("   Agent names found:", agent_names)
        else:
            print("✗ Team does not have '_participants' attribute")

    except Exception as e:
        print(f"✗ Failed to create team: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
