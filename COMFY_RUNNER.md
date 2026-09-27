# Comfy Runner

1. In ComfyUI, export your workflow in **API format**. The normal frontend workflow JSON contains canvas state and is not accepted.
2. Create a dataset JSON with an explicit value for every run:

   ```json
   {
     "format": "potatoforge_comfy_dataset",
     "version": 1,
     "name": "Calibration sample",
     "runs": [
       {"prompt": "A portrait in daylight", "seed": 104827361, "resolution": 0.7},
       {"prompt": "A railway terminal", "seed": 773510924, "resolution": 0.7}
     ]
   }
   ```

3. Open **Comfy Runner** in PotatoForge. Enter the ComfyUI URL and connect, or provide a local `.bat`/`.cmd` file and working directory to launch it.
4. Enter a folder containing JSON files and choose the API workflow and dataset, or enter their full paths as seen by the PotatoForge backend. Inspect each file, then bind dataset fields to literal workflow inputs. One field may bind to several inputs.
5. Run **Check setup**, then **Start sequential run**. The backend rechecks the files and connection before starting. It freezes their contents for this run.
6. Watch each row in the run table. It shows the seed, Comfy prompt ID, current node, cached nodes, and any error. A failed row stops later rows. Cancel targets the current PotatoForge prompt when Comfy supports job cancellation.

Seeds are explicit non-negative integers up to `2^64 - 1`. `resolution`, when present, must be between `0` and `1`. Other primitive JSON fields can be bound. Browser refreshes reconnect to the backend job; a PotatoForge backend restart marks an interrupted job failed so it is not mistaken for a successful calibration.

GPTQ workflows that need a separate finalization execution can use the **Optional final workflow JSON path**. That API workflow runs once after all dataset rows succeed. Use a finalization workflow that explicitly names the same calibration session.

For activation calibration, put one `PotatoForgeActivationCalibration` and one `PotatoForgeFinalizeActivationCalibration` node in the dataset workflow. Set a stable session name and model identity. Comfy Runner starts the session on the first row, resumes it on later rows, commits each successful row, and finalizes it on the last row. If a run fails, the earlier rows remain in the working session. To resume explicitly, set the calibration node's `session_action` to `resume` and run a dataset containing only the remaining rows. The final JSON/Safetensors pair can go straight to `activation audit` and `activation optimize`.
