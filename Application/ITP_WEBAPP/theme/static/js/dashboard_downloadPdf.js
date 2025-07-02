// static/js/dashboard_downloadPdf.js

document.addEventListener('DOMContentLoaded', function() {
    const downloadPdfBtn = document.getElementById('download-pdf-btn');

    if (downloadPdfBtn) {
        downloadPdfBtn.addEventListener('click', function () {
            const element = document.getElementById('data-table'); // The table element to convert

            // Get the video title from the data attribute
            let videoTitle = downloadPdfBtn.dataset.videoTitle || 'video_data_default';

            // --- LOGIC TO REMOVE .mp4 ---
            let lowerCaseVideoTitle = videoTitle.toLowerCase();
            if (lowerCaseVideoTitle.endsWith('mp4')) {
                videoTitle = videoTitle.substring(0, videoTitle.length - 3);
            }
            // --- END LOGIC ---

            // Ensure it ends with .pdf
            if (!videoTitle.endsWith('.pdf')) {
                videoTitle += '.pdf';
            }

            html2pdf().from(element).set({
                margin: [0.5, 0.5, 0.5, 0.5], // Adjust margins (top, left, bottom, right) for more space
                filename: videoTitle,
                html2canvas: {
                    scale: 0.8, // Adjust this value to scale down the content (e.g., 0.8 for 80%)
                    // You might need to experiment with this value.
                    // If the table is still too wide, try a smaller scale like 0.7 or 0.6.
                },
                jsPDF: {
                    unit: 'in',
                    format: 'letter', // You can try 'a4' or 'legal' or even 'a3'
                    orientation: 'landscape', // Set to landscape for wider tables
                }
            }).save();
        });
    }
});