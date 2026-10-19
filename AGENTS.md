# Runpod account for this repository

Use only the Runpod API or MCP. All pods and volumes for this repository must
belong to the **Caleb SPAR** team, never the Clay personal account.

- Read the API key from
  `~/.config/runpod/accounts/verbalizing-subliminal-learning/api_key` directly
  into the client's `Authorization: Bearer ...` header. Never print or copy it
  into this repository or logs.
- Send `X-Team-Id: cmu31hyiy002gk9s0xik85gob` on every API request. This key
  otherwise defaults to the Clay personal account.
- Before creating resources, query GraphQL `myself { id email }` with both
  headers. Require account ID `user_3JNPrVMTYwugXaYJytSv2t8lRsv`
  (`caleb.d.biddulph+spar@gmail.com`). Fail closed on a mismatch.
- Do not fall back to `~/.config/runpod/api_key`, ambient credentials, or an
  MCP connection whose team scope has not been verified. Do not change global
  Runpod settings.
- GPU capacity is scarce. Query live GPU availability, create the network
  volume in the available GPU's data center, and immediately create the pod
  with that volume attached. If capacity disappears, remove only the newly
  created empty volume from that unsuccessful attempt before trying elsewhere.
- Discover public IPs and SSH ports through the API; for REST v1, the direct
  SSH port is `portMappings["22"]`.

## Provisioned resources (2026-10-19)

- Pod: `dsln6o9uc00dwc`, `verbalizing-subliminal-learning`
- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB VRAM
- Data center: `US-NE-1`
- Network volume: `ae6w0yoyy6`, 100 GB, mounted at `/workspace`

Recheck resource status through the API before relying on these details.

# Report links

When sharing HTML reports, write the full absolute filesystem path as plain
text on its own line. A clickable link may be added separately.
