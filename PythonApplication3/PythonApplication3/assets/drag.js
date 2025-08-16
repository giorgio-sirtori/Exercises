document.addEventListener('DOMContentLoaded', function () {
    let draggedId = null;

    // Handle drag start
    document.body.addEventListener('dragstart', function (e) {
        if (e.target && e.target.classList.contains('draggable')) {
            draggedId = e.target.id;
            e.dataTransfer.setData('text/plain', draggedId);
        }
    });

    // Handle drag over on drop zone
    document.body.addEventListener('dragover', function (e) {
        if (e.target && e.target.dataset.droppable === 'true') {
            e.preventDefault();
        }
    });

    // Handle drop
    document.body.addEventListener('drop', function (e) {
        if (e.target && e.target.dataset.droppable === 'true') {
            e.preventDefault();
            const id = e.dataTransfer.getData('text/plain');

            // Push data to Dash Store
            if (window.dash_clientside) {
                const storeComponent = window.dash_clientside.getComponent('dragged-store');
                if (storeComponent) {
                    storeComponent.setProps({ data: { id: id } });
                }
            }
        }
    });
});
