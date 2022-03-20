import result_json from "./results/result_refine.js";
import order_json from "./order.js";
import test_queries from "./test_queries.js"

const button_input = document.getElementsByClassName("btn-input")[0];

const search_element = document.getElementById("text-input")
createOption(search_element, test_queries)

button_input.addEventListener("click", () => {
    const search_key = search_element.options[search_element.selectedIndex].value
    createOutput(result_json, order_json, search_key)
});

search_element.addEventListener('change', () => {
    const search_key = search_element.options[search_element.selectedIndex].value
    const caption_element = document.getElementsByClassName("caption")[0];
    if (test_queries.hasOwnProperty(search_key)) {
        caption_element.innerHTML = test_queries[search_key].join("<br>")
    }
    else {
        caption_element.innerHTML = ""
        const outputDisplay = document.getElementsByClassName("video-output")[0];
        outputDisplay.innerHTML = ""
    }
});

function createOption(search_element, test_queries){
    var list_keys = Object.keys(test_queries)
    for (let i=0; i<list_keys.length; i++) {
        var key_element = document.createElement("option")
        key_element.setAttribute(
            "value",
            list_keys[i]
        )
        key_element.innerHTML = `${list_keys[i]}-${i+1}`
        // key_element.innerHTML = `${list_keys[i]}`

        search_element.appendChild(key_element)
    }
}

function createOutput(result_json, order_json, search_key){
    const outputDisplay = document.getElementsByClassName("video-output")[0];
    outputDisplay.innerHTML = ""
    var output = result_json[search_key]
    for (let i=0; i< 21; i++) {
        var vidName = order_json[output[i]]
        var vidNameDisplay = `${i+1}-${vidName}`
        var container = document.createElement("div")
        container.setAttribute(
            "class",
            "container"
        )
        var video = document.createElement("video")
        video.playbackRate = 3.0
        video.setAttribute(
            "controls",
            "true"
        )
        video.setAttribute(
            "class",
            "video-element"
        )
        video.setAttribute(
            "autoplay",
            "true"
        )
        video.setAttribute(
            "loop",
            "true"
        )
        var source = document.createElement("source");
        source.setAttribute(
            "type",
            "video/mp4"
        );
        source.setAttribute(
            "src",
            `./video_mp4/${vidName}.mp4`
        );
        video.appendChild(source);
        var nameDisplay = document.createElement("p")
        nameDisplay.setAttribute(
            "class",
            "video-name"
        )
        nameDisplay.innerHTML=vidNameDisplay;

        container.appendChild(video)
        container.appendChild(nameDisplay)
        outputDisplay.appendChild(container)
    }
}
