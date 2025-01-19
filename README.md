# BlenderCs2ExporterPlugin
A plugin for Blender that allows to export .cs2 files (intermediate files in Total War assets processing pipeline)

# Blender CS2 exporter plugin (WORK IN PROGRESS)
The only software that has official plugins provided by CA to export .cs2 files is 3ds max, maya and sometimes motion builder. These are all proprietary software that costs money. For many modders it means that they either have to spend money on pricey applications or pirate the software. Neither of which is a great option and severely limits entry level to new modders.

Blender, however, is a free software that is easily available to everyone. The idea behind this project is to create a .cs2 exporter plugin for Blender.

The project is currently work in progress and It's stuck in limbo due to inability to create custom fx shader in Blender. Sadly, Blender only provides the ability to use pre-made shaders. For exporting cs2 files we need a custom shader. We don't even need Blender to render materials identical to TW games but we need it to be able to properly export material parameters, such as texture slots, shader technique, material attributes, etc etc.

Credits for the entire project: @victimized.
