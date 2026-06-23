document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('cardsContainer');
    const searchInput = document.getElementById('searchInput');
    const validOnlyToggle = document.getElementById('validOnlyToggle');
    const totalCount = document.getElementById('totalCount');

    // Debounce function for search
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    function renderCards() {
        const searchTerm = searchInput.value.toLowerCase();
        const showOnlyValid = validOnlyToggle.checked;

        // Filter cards
        const filteredCards = CARDS_DATA.filter(card => {
            // Check validity filter
            if (showOnlyValid && !card.is_valid) return false;

            // Check search term
            if (searchTerm) {
                const searchString = `${card.id} ${card.name} ${card.type} ${card.stage} ${card.attack1_name}`.toLowerCase();
                if (!searchString.includes(searchTerm)) return false;
            }

            return true;
        });

        // Update count
        totalCount.textContent = filteredCards.length;

        // Clear container
        container.innerHTML = '';

        // If no results
        if (filteredCards.length === 0) {
            container.innerHTML = `
                <div style="grid-column: 1 / -1; text-align: center; padding: 3rem; color: var(--text-secondary);">
                    <h3>No cards found matching your criteria.</h3>
                </div>
            `;
            return;
        }

        // We will render only the first 100 for performance unless narrowed down, 
        // but with 200 cards max usually (valid), it's fine. 
        // Max cards to render to prevent DOM freeze
        const maxRender = 150;
        const cardsToRender = filteredCards.slice(0, maxRender);

        // Build HTML
        const html = cardsToRender.map((card, index) => {
            const validBadge = card.is_valid 
                ? '<span class="badge badge-valid">CABT Valid</span>' 
                : '<span class="badge badge-invalid">Invalid</span>';

            const energyCost = card.attack1_cost && card.attack1_cost !== 'n/a' 
                ? card.attack1_cost 
                : 'Free';

            const attackDmg = card.attack1_dmg && card.attack1_dmg !== 'n/a' 
                ? card.attack1_dmg 
                : '';

            return `
                <div class="poke-card fade-in" style="animation-delay: ${(index % 10) * 0.05}s">
                    <div class="card-header">
                        <div class="card-title">
                            <span class="card-id">#${card.id} | ${card.set}</span>
                            <h2 class="card-name">${card.name}</h2>
                        </div>
                        <div style="text-align: right;">
                            <span class="card-hp">HP ${card.hp !== 'n/a' ? card.hp : '-'}</span>
                        </div>
                    </div>
                    
                    <div class="card-meta">
                        ${validBadge}
                        <span class="meta-pill">${card.stage}</span>
                        <span class="meta-pill">Type: ${card.type !== 'n/a' ? card.type : 'None'}</span>
                        <span class="meta-pill">Retreat: ${card.retreat !== 'n/a' ? card.retreat : '0'}</span>
                    </div>

                    ${card.attack1_name && card.attack1_name !== 'n/a' ? `
                    <div class="card-attack">
                        <div class="attack-header">
                            <span>🗡️ ${card.attack1_name}</span>
                            <span>${energyCost} ${attackDmg ? '| 💥 ' + attackDmg : ''}</span>
                        </div>
                        <p class="attack-desc">${card.attack1_desc !== 'n/a' ? card.attack1_desc : 'No additional effects.'}</p>
                    </div>
                    ` : '<div style="margin-top:auto; text-align:center; color: var(--text-secondary); font-size: 0.9rem;">No attacks</div>'}
                </div>
            `;
        }).join('');

        container.innerHTML = html;

        if (filteredCards.length > maxRender) {
            container.innerHTML += `
                <div style="grid-column: 1 / -1; text-align: center; padding: 2rem; color: var(--text-secondary);">
                    <p>Showing ${maxRender} of ${filteredCards.length} cards. Please refine your search to see more.</p>
                </div>
            `;
        }
    }

    // Event Listeners
    searchInput.addEventListener('input', debounce(renderCards, 300));
    validOnlyToggle.addEventListener('change', renderCards);

    // Initial render
    renderCards();
});
