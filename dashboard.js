// Dashboard specific functionality
document.addEventListener('DOMContentLoaded', function() {
    loadProfiles();
});

function loadProfiles() {
    // Simulated profile data
    const profiles = [
        {
            id: '1',
            name: 'John Doe',
            status: 'active',
            dateOfBirth: '1985-06-15',
            lastUpdated: '2023-12-20'
        },
        // Add more profile data as needed
    ];

    const profileGrid = document.getElementById('profile-grid');
    profiles.forEach(profile => {
        const card = createProfileCard(profile);
        profileGrid.appendChild(card);
    });
}

function createProfileCard(profile) {
    const card = document.createElement('div');
    card.className = 'bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow cursor-pointer';
    card.innerHTML = `
        <div class="flex justify-between items-start mb-4">
            <h3 class="text-xl font-semibold text-gray-900">${profile.name}</h3>
            <span class="px-2 py-1 rounded-full text-sm ${getStatusColor(profile.status)}">
                ${profile.status}
            </span>
        </div>
        <div class="space-y-2 text-sm text-gray-600">
            <p>DOB: ${formatDate(profile.dateOfBirth)}</p>
        </div>
        <div class="mt-4 pt-4 border-t border-gray-200">
            <p class="text-sm text-gray-500">
                Last updated: ${formatDate(profile.lastUpdated)}
            </p>
        </div>
    `;
    return card;
}

function getStatusColor(status) {
    const colors = {
        active: 'bg-red-100 text-red-800',
        arrested: 'bg-green-100 text-green-800',
        deceased: 'bg-gray-100 text-gray-800',
        unknown: 'bg-yellow-100 text-yellow-800'
    };
    return colors[status] || colors.unknown;
}

function formatDate(dateString) {
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
}