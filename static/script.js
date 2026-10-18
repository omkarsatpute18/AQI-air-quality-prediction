const form = document.getElementById("predictionForm");

const loading = document.getElementById("loading");

const result = document.getElementById("result");

const errorMessage =
    document.getElementById("errorMessage");

const aqiValue =
    document.getElementById("aqiValue");

const aqiCategory =
    document.getElementById("aqiCategory");



form.addEventListener("submit", async function (event) {

    event.preventDefault();


    // Hide previous result/error
    result.classList.add("hidden");

    errorMessage.classList.add("hidden");

    loading.classList.remove("hidden");


    // Collect input values
    const data = {

        "PM2.5":
            parseFloat(
                document.getElementById("PM25").value
            ),

        "PM10":
            parseFloat(
                document.getElementById("PM10").value
            ),

        "NO":
            parseFloat(
                document.getElementById("NO").value
            ),

        "NO2":
            parseFloat(
                document.getElementById("NO2").value
            ),

        "NOx":
            parseFloat(
                document.getElementById("NOx").value
            ),

        "NH3":
            parseFloat(
                document.getElementById("NH3").value
            ),

        "CO":
            parseFloat(
                document.getElementById("CO").value
            ),

        "SO2":
            parseFloat(
                document.getElementById("SO2").value
            ),

        "O3":
            parseFloat(
                document.getElementById("O3").value
            ),

        "Benzene":
            parseFloat(
                document.getElementById("Benzene").value
            ),

        "Toluene":
            parseFloat(
                document.getElementById("Toluene").value
            ),

        "Xylene":
            parseFloat(
                document.getElementById("Xylene").value
            )
    };


    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify(data)
            }
        );


        const resultData =
            await response.json();


        if (!response.ok || !resultData.success) {

            throw new Error(
                resultData.error ||
                "Prediction failed."
            );

        }


        // Display prediction
        aqiValue.textContent =
            resultData.aqi;


        aqiCategory.textContent =
            resultData.category;


        result.classList.remove("hidden");


        // Scroll to result
        result.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });


    } catch (error) {

        errorMessage.textContent =
            error.message ||
            "Something went wrong while making the prediction.";

        errorMessage.classList.remove("hidden");

    } finally {

        loading.classList.add("hidden");

    }

});



/* Reset button */

form.addEventListener("reset", function () {

    result.classList.add("hidden");

    errorMessage.classList.add("hidden");

});