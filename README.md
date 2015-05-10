Photo-Mosaic Builder made in Python by Danny Shum

Execution: To use this code, execute a command structured like this:   Python [program name] "[image file name]" [number of tiles]

Restrictions: 
- The number of tiles cannot exceed 9800. 
- You must have a library of images available in the same directory.
- You must first build a library of images. You can do this by changing the command in the __main__ function from "loadLib(...)" to "buildLib(...)". I may change this in the future to be more user friendly.

Notes: 
- This program works best for large images. I'd recommend your image exceed 2000x2000 pixels at least, though it doesn't need to. 
- Program runtime increases with the number of tiles requested. A 3000x3000 image broken into ~6000 tiles may take 10-20 minutes. (shhh, Art takes time.)
