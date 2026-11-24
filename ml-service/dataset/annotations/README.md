# Image manifest

Use `image_manifest_template.csv` to record provenance for generated images. Keep `dataset_label` set to `synthetic` and do not enter invented tape measurements. Marker dimensions must remain unverified unless checked against the physical printed marker.

`image_rename_map.csv` records the original filename corresponding to each numbered image. Image files alone do not train the MUAC Ridge corrector: that requires a verified tape MUAC value paired with each photograph. A learned arm segmentation model requires a manually reviewed arm mask for each training image. Keep both label types separate from synthetic-only examples.
