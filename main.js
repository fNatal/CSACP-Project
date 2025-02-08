// Common functionality
document.addEventListener('DOMContentLoaded', function() {
    // Load navigation and footer
    const navPlaceholder = document.getElementById('nav-placeholder');
    const footerPlaceholder = document.getElementById('footer-placeholder');

    if (navPlaceholder) {
        fetch('/components/navigation.html')
            .then(response => response.text())
            .then(data => {
                navPlaceholder.innerHTML = data;
                highlightCurrentPage();
            });
    }

    if (footerPlaceholder) {
        fetch('/components/footer.html')
            .then(response => response.text())
            .then(data => {
                footerPlaceholder.innerHTML = data;
            });
    }
});

function highlightCurrentPage() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('nav a');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('bg-indigo-700', 'text-white');
            link.classList.remove('text-indigo-100', 'hover:bg-indigo-500');
        }
    });
}