// This macro will calculate the percentage of colocalization of img2 marker in img1

open("img1.tif");
open("img2.tif");

imageCalculator("AND create", "img1.tif","img2.tif");
selectWindow("Result of img1.tif");

selectWindow("img2.tif");
getHistogram(values, counts, 256);
img2_pixels = (counts[255]);

selectWindow("Result of img1.tif");
getHistogram(values, counts, 256);
colocalization_pixels = (counts[255]);
print("number of pixels in both img1 & img2 = ", colocalization_pixels)
print("number of pixels in img2 = ", img2_pixels)
print("percentage of colocalization = ", (colocalization_pixels / img2_pixels) * 100)