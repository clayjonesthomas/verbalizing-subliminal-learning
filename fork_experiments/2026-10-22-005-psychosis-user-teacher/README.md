# Psychosis teacher with instructions in user message

Results pending. Regenerate the first 1,000 training inputs and all 100 validation / 100 test inputs with the existing pinned psychosis RL teacher. Move the original system text into the user message and leave the teacher system message empty. This aligns the textual context with the proposed 256-embedding-only student system message.

Four independent RTX PRO 6000 96 GB workers, deterministic ID sharding, original sampling settings. Retain every attempt and retry only format/length failures until all 1,200 original IDs have accepted outputs. No behavioral selection. Preserve failed workers for recovery; retire successful workers only after artifacts and environment are saved on the network volume.
