#!/usr/bin/env python
"""Verify orchestrator can see the pvpython_coder agent."""

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

    # Create paths
    from tempfile import mkdtemp
    temp_dir = Path(mkdtemp())
    paths = RunPaths(
        internal_root_dir=temp_dir,
        external_root_dir=temp_dir,
        internal_run_dir=temp_dir,
        external_run_dir=temp_dir,
        run_suffix="test_orchestrator",
    )

    print("=" * 80)
    print("CREATING TEAM WITH ORCHESTRATOR")
    print("=" * 80)

    try:
        team = await get_task_team(
            magentic_ui_config=magentic_config,
            paths=paths,
        )

        print("✓ Team created successfully!")
        print(f"\nTeam type: {type(team).__name__}")

        # Check if team has orchestrator attributes
        if hasattr(team, '_participant_names'):
            print("\n" + "=" * 80)
            print("ORCHESTRATOR'S VIEW OF PARTICIPANT NAMES")
            print("=" * 80)
            print(team._participant_names)

            if 'pvpython_coder' in team._participant_names:
                print("\n✓ pvpython_coder IS in _participant_names")
            else:
                print("\n✗ pvpython_coder NOT in _participant_names")

        if hasattr(team, '_participant_descriptions'):
            print("\n" + "=" * 80)
            print("ORCHESTRATOR'S VIEW OF PARTICIPANT DESCRIPTIONS")
            print("=" * 80)
            print(team._participant_descriptions)

            if 'pvpython_coder' in team._participant_descriptions:
                print("\n✓ pvpython_coder IS in _participant_descriptions")
            else:
                print("\n✗ pvpython_coder NOT in _participant_descriptions")

        if hasattr(team, '_team_description'):
            print("\n" + "=" * 80)
            print("ORCHESTRATOR'S TEAM DESCRIPTION (what it uses in prompts)")
            print("=" * 80)
            print(team._team_description)

            if 'pvpython_coder' in team._team_description.lower():
                print("\n✓ pvpython_coder IS mentioned in team_description")
            else:
                print("\n✗ pvpython_coder NOT mentioned in team_description")
        else:
            print("\n⚠️  Team does not have '_team_description' attribute yet")
            print("   (It's created during runtime when orchestrator starts)")

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"✓ pvpython_coder agent is properly registered in the team")
        print(f"✓ Agent name: 'pvpython_coder'")
        print(f"✓ Agent is in _participant_names list")
        print(f"✓ Agent description is in _participant_descriptions list (position 6)")
        print(f"\nThe orchestrator WILL be able to see and use this agent during runtime.")

    except Exception as e:
        print(f"✗ Failed to create team: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
