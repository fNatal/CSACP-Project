// Analytics specific functionality
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    loadStats();
});

function initializeCharts() {
    const ctx = document.getElementById('profileChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Active', 'Arrested', 'Deceased', 'Unknown'],
            datasets: [{
                label: 'Profile Status Distribution',
                data: [12, 19, 3, 5],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.2)',
                    'rgba(16, 185, 129, 0.2)',
                    'rgba(107, 114, 128, 0.2)',
                    'rgba(245, 158, 11, 0.2)'
                ],
                borderColor: [
                    'rgba(239, 68, 68, 1)',
                    'rgba(16, 185, 129, 1)',
                    'rgba(107, 114, 128, 1)',
                    'rgba(245, 158, 11, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function loadStats() {
    const statsContainer = document.querySelector('.grid');
    const stats = [
        { title: 'Total Profiles', value: 39 },
        { title: 'Active Cases', value: 12, trend: 'up' },
        { title: 'Arrested', value: 19, trend: 'down' },
        { title: 'Unknown Status', value: 5 }
    ];

    stats.forEach(stat => {
        const card = createStatCard(stat);
        statsContainer.appendChild(card);
    });
}

function createStatCard({ title, value, trend }) {
    const card = document.createElement('div');
    card.className = 'bg-white overflow-hidden shadow rounded-lg';
    card.innerHTML = `
        <div class="p-5">
            <div class="flex items-center">
                <div class="flex-1">
                    <p class="text-sm font-medium text-gray-500 truncate">${title}</p>
                    <p class="mt-1 text-3xl font-semibold text-gray-900">${value}</p>
                </div>
                ${trend ? getTrendIcon(trend) : ''}
            </div>
        </div>
    `;
    return card;
}

function getTrendIcon(trend) {
    const color = trend === 'up' ? 'text-green-600' : 'text-red-600';
    const path = trend === 'up' 
        ? 'M5 10l7-7m0 0l7 7m-7-7v18' 
        : 'M19 14l-7 7m0 0l-7-7m7 7V3';
    
    return `
        <div class="flex items-center ${color}">
            <svg class="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path d="${path}"/>
            </svg>
        </div>
    `;
}