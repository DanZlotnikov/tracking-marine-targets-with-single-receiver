# Bing Map Tile Download
The functions download_tiles and stitch_tiles based on the project "Bing Map Tile Download" <br>
https://github.com/dakshaau/map_tile_download


**Input:** Top Latitude, Top Longitude, Bottom Latitude, Bottom Longitude

**Output:** Image bounded by Latitude Longitude rectangle

### Description of the main functions

1. download_map: <br> downloads the relevant map tiles from Bing servers. <br> **NOTE:** The maximum image resolution can be 20,512 X 20,512 pixels. If the maximum possible resolution
          for the complete image is greater than this, the code will discard the lat long pairs.

```Batchfile
download_map(<top_lat>, <top_long>, <bottom_lat>, <bottom_long>)
```
2. stitchTiles: <br> After all the tiles have been downloaded, run "stitchTiles" to merge files
and then crop it to fit the bounding rectangle

```Batchfile
stitchTiles()
```
