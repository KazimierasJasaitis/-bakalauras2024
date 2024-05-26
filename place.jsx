// Define the path to the output.txt file
var scriptFile = new File($.fileName);
var scriptPath = scriptFile.path; // Gets the path of the currently running script
var folder1 = prompt("Enter the first folder name:", "3");
var folder2 = prompt("Enter the second folder name:", "301");
var folderPath = scriptPath + "/" + folder1 + "/" + folder2 + "/output.txt";

var inputFile = new File(folderPath);

function parseScientificNotation(str) {
    var value = parseFloat(str);
    if (!isFinite(value)) {
        alert("Failed to parse float from: " + str);
    }
    return value;
}

if (inputFile.open("r")) {
    var docWidth = 100; // Set width of the document
    var docHeight = 100; // Set height of the document
    var doc = app.documents.add(docWidth, docHeight, 72, 'output', NewDocumentMode.RGB, DocumentFill.TRANSPARENT);

    var idx = 1; // Start index for image files

    while (!inputFile.eof) {
        var line = inputFile.readln();
        var cleanedLine = line.replace(/[\[\]]/g, '').replace(/^\s+|\s+$/g, ''); // Remove brackets and trim spaces
        var values = cleanedLine.split(/\s+/); // Split by any whitespace

        if (values.length >= 2) { // Ensure there are at least three parts
            var x = 0; // Set X to 0 as per example
            var y = Math.round(parseScientificNotation(values[1])); // Round Y value
            var scale = parseScientificNotation(values[2]) * 100; // Calculate scale percentage

            var imageFileName = scriptPath + "/" + folder1 + "/" + folder2 + "/" + (idx < 10 ? '0' : '') + idx + ".png";
            var imageFile = new File(imageFileName);

            if (imageFile.exists) {
                app.open(imageFile);
                var imageDoc = app.activeDocument;
                imageDoc.selection.selectAll();
                imageDoc.selection.copy();
                imageDoc.close(SaveOptions.DONOTSAVECHANGES);

                app.activeDocument = doc;
                doc.paste();

                var imageLayer = doc.activeLayer;

                // Scale the image
                imageLayer.resize(scale, scale, AnchorPosition.TOPCENTER);  // Ensure scaling does not affect the y-coordinate placement

                // Adjust positioning to place the top of the image at the top of the document
                imageLayer.translate(-imageLayer.bounds[0], -imageLayer.bounds[1]); // Reset position to top left corner
                imageLayer.translate(x, y); // Apply the y translation to place the top of the image accordingly

                // Output the coordinates for debugging
                alert("Placing image " + idx + " at X: " + x + ", Y: " + y + ", Scale: " + scale);

                idx++;
            } else {
                alert("Image file does not exist: " + imageFileName);
            }
        } else {
            alert("Incorrect data format in line: " + line);
        }
    }

    inputFile.close();
    alert("Images have been placed successfully.");
} else {
    alert("Failed to open file: " + folderPath);
}
