const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

const grid = 20;
let count = 0;
let snake = [{ x: 160, y: 160 }, { x: 140, y: 160 }, { x: 120, y: 160 }, { x: 100, y: 160 }];
let direction = 'right';
let food = {
    x: Math.floor(Math.random() * 20) * grid,
    y: Math.floor(Math.random() * 20) * grid
};

function update() {
    if (++count < 4) {
        return;
    }
    count = 0;

    // Move snake by adding a new head
    const head = { ...snake[0] };

    if (direction === 'left') head.x -= grid;
    if (direction === 'right') head.x += grid;
    if (direction === 'up') head.y -= grid;
    if (direction === 'down') head.y += grid;

    snake.unshift(head);

    // Check if the snake eats the food
    if (head.x === food.x && head.y === food.y) {
        food.x = Math.floor(Math.random() * 20) * grid;
        food.y = Math.floor(Math.random() * 20) * grid;
    } else {
        snake.pop();
    }

    // Check for collision with walls
    if (head.x < 0 || head.x >= canvas.width || head.y < 0 || head.y >= canvas.height) {
        resetGame();
    }

    // Check for collision with itself
    for (let i = 4; i < snake.length; i++) {
        if (head.x === snake[i].x && head.y === snake[i].y) {
            resetGame();
        }
    }
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw the snake
    ctx.fillStyle = 'lime';
    snake.forEach(segment => ctx.fillRect(segment.x, segment.y, grid, grid));

    // Draw the food
    ctx.fillStyle = 'red';
    ctx.fillRect(food.x, food.y, grid, grid);
}

function resetGame() {
    snake = [{ x: 160, y: 160 }, { x: 140, y: 160 }, { x: 120, y: 160 }, { x: 100, y: 160 }];
    direction = 'right';
    food = {
        x: Math.floor(Math.random() * 20) * grid,
        y: Math.floor(Math.random() * 20) * grid
    };
}

function changeDirection(event) {
    if (event.key === 'ArrowLeft' && direction !== 'right') direction = 'left';
    if (event.key === 'ArrowRight' && direction !== 'left') direction = 'right';
    if (event.key === 'ArrowUp' && direction !== 'down') direction = 'up';
    if (event.key === 'ArrowDown' && direction !== 'up') direction = 'down';
}

document.addEventListener('keydown', changeDirection);

function gameLoop() {
    update();
    draw();
    requestAnimationFrame(gameLoop);
}

requestAnimationFrame(gameLoop);
