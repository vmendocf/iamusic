//py -m http.server 8001

let model;

document.addEventListener('DOMContentLoaded', 
async function loadModel() {
	console.log("model loading..");
	loader = document.getElementById("progress-box");
	load_button = document.getElementById("load-button");
	loader.style.display = "block";
	modelName = "pre-trained-vgg";
	model = undefined;
	model = await tf.loadLayersModel('../output/web_model/model.json');
	loader.style.display = "none";
	load_button.disabled = true;
	load_button.innerHTML = "Loaded Model";
	console.log("model loaded..");
}
);



// async function loadModel() {
// 	console.log("model loading..");
// 	loader = document.getElementById("progress-box");
// 	load_button = document.getElementById("load-button");
// 	loader.style.display = "block";
// 	modelName = "pre-trained-vgg";
// 	model = undefined;
// 	model = await tf.loadLayersModel('../output/web_model/model.json');
// 	loader.style.display = "none";
// 	load_button.disabled = true;
// 	load_button.innerHTML = "Loaded Model";
// 	console.log("model loaded..");
// }

async function loadFile() {
	console.log("image is in loadfile..");
	document.getElementById("select-file-box").style.display = "table-cell";
  	document.getElementById("predict-box").style.display = "table-cell";
  	// document.getElementById("prediction").innerHTML = "Click predict to find the class image!";
  	var fileInputElement = document.getElementById("select-file-image");
  	console.log(fileInputElement.files[0]);
    renderImage(fileInputElement.files[0]);
}

function renderImage(file) {
  var reader = new FileReader();
  console.log("image is here..");
  reader.onload = function(event) {
    img_url = event.target.result;
    console.log("image is here2..");
    document.getElementById("test-image").src = img_url;
  }
  reader.readAsDataURL(file);
}

const CATEGORY_CLASSES = {
  0: 'Flûte de pan',
  1: 'Flûte traversière',
  2: 'Harmonica'
};

async function predButton() {
	console.log("model loading..");

	if (model == undefined) {
		alert("Please load the model first..")
	}
	if (document.getElementById("predict-box").style.display == "none") {
		alert("Please load an image using 'Demo Image' or 'Upload Image' button..")
	}
	console.log(model);
	let image  = document.getElementById("test-image");
	let tensor = preprocessImage(image, modelName);

	let predictions = await model.predict(tensor).data();
	console.log('prediction:::'+ predictions);

	let results = Array.from(predictions)
		.map(function (p, i) {
			return {
				probability: p,
				className: CATEGORY_CLASSES[i]
			};
		}).sort(function (a, b) {
			return b.probability - a.probability;
		}).slice(0, 3);

	document.getElementById("predict-box").style.display = "block";

	if(results[0].className == "Harmonica"){
		console.log('harmonica');
		document.getElementById("prediction").innerHTML = '<a href="https://www.youtube.com/watch?v=qWLvv_rrSH4" target="_blank" class="link">Découvrir l\' '+ results[0].className + ' en cliquant ici</a>';

	}else if(results[0].className == "Flûte de pan"){
		console.log('panpan');
		document.getElementById("prediction").innerHTML = '<a href="https://www.youtube.com/watch?v=xyoco3hG_5k" target="_blank" class="link">Découvrir la '+ results[0].className + ' en cliquant ici</a>';

		// window.open('https://www.youtube.com/watch?v=xyoco3hG_5k', '_blank');
		
	}
	var ul = document.getElementById("predict-list");
	ul.innerHTML = "";
	results.forEach(function (p) {
		console.log(p.className + " " + p.probability.toFixed(6));
		var li = document.createElement("LI");
		li.innerHTML = p.className + " " + p.probability.toFixed(6);
		ul.appendChild(li);
	});

}

function preprocessImage(image, modelName) {
	let tensor = tf.browser.fromPixels(image)
		.resizeNearestNeighbor([224, 224])
		.toFloat();

	if (modelName === "undefined") {
		aa = tensor.expandDims();
		console.log(aa.shape)
		return tensor.expandDims();
	} else if (modelName === "pre-trained-vgg") {
		let offset = tf.scalar(255.0);
		return tensor.sub(offset)
			.div(offset)
			.expandDims();
	} else if (modelName === "mobilenet") {
		let offset = tf.scalar(127.5);
		return tensor.sub(offset)
			.div(offset)
			.expandDims();
	}
	else if(modelName=="vgg")
    {
        let meanImageNetRGB= tf.tensor1d([123.68,116.779,103.939]);
        return tensor.sub(meanImageNetRGB)
                    .reverse(2)   // using the conventions from vgg16 documentation
                    .expandDims();
          console.log("inside the vgg preProcessing:");
    } else {
		alert("Unknown model name..")
	}
}

// function loadDemoImage() {
// 	document.getElementById("predict-box").style.display = "table-cell";
//   	document.getElementById("prediction").innerHTML = "Click predict to find my label!";
// 	document.getElementById("select-file-box").style.display = "table-cell";
// 	document.getElementById("predict-list").innerHTML = "";

// 	base_path = "dataset/test/unknown/unnamed.jpg"
// 	// maximum = 4;
// 	// minimum = 1;
// 	// var randomnumber = Math.floor(Math.random() * (maximum - minimum + 1)) + minimum;
// 	// img_path = base_path + randomnumber + ".jpeg"
// 	img_path = base_path
// 	document.getElementById("test-image").src = img_path;
// }
