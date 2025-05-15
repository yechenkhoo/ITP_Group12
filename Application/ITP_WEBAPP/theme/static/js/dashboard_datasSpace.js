
function openTab(event, tabId) {
    var i, tabcontent, tablinks;
    
    // Hide all tab contents
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].classList.add("hidden");
    }

    // Remove active class from all tab buttons
    tablinks = document.querySelectorAll("[data-tab-target]");
    for (i = 0; i < tablinks.length; i++) {
    tablinks[i].classList.remove("border-blue-500", "text-blue-500");
    tablinks[i].classList.add("border-transparent", "hover:text-gray-600", "hover:border-gray-300");
    }

    // Show the current tab content
    document.getElementById(tabId).classList.remove("hidden");

    // Add active class to the current tab button
    event.currentTarget.classList.add("border-blue-500", "text-blue-500");
    event.currentTarget.classList.remove("border-transparent", "hover:text-gray-600", "hover:border-gray-300");
}
