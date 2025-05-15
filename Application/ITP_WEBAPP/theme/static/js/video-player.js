document.addEventListener("DOMContentLoaded", function () {
    const videoPlayer = document.getElementById("video-player");
    const rows = document.querySelectorAll("tbody tr");

    rows.forEach(row => {
        row.addEventListener("click", function () {
            const timestamp = parseFloat(row.dataset.timestamp); // Get timestamp from data attribute
            if (!isNaN(timestamp) && videoPlayer) {
                videoPlayer.currentTime = timestamp; // Seek to the timestamp
            }
        });
    });
});