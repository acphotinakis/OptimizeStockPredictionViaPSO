
Particle Swarm Optimization (PSO) is utilized by the researchers as a swarm-intelligence mechanism to automatically search for and identify the optimal hyperparameter configurations for Long Short-Term Memory (LSTM) neural networks
. Because LSTM models are highly sensitive to parameter selection, relying on subjective human experience to guess the optimal network architecture often yields poor predictive performance in highly chaotic, nonlinear stock markets
.
By employing PSO, the researchers treat different candidate combinations of LSTM hyperparameters as "particles" flying through a multi-dimensional search space
. Each particle evaluates its predictive accuracy (fitness) and shares that information with the swarm, dynamically updating its velocity and position until the swarm collectively converges on the absolute best network configuration
.
Of the four provided studies, three actively integrate and improve upon PSO, while one (Lanbouri & Achchab) relies on a manually configured baseline LSTM. Here is exactly how each team utilizes the algorithm.
1. Zeng et al.: Standard PSO for Deep Architectures
Zeng et al. utilize the standard PSO algorithm specifically to navigate the non-convex, high-dimensional problem of tuning multi-layer LSTM networks
.
The Search Space: The researchers define particles to search for the optimal number of training iterations (epochs) and the optimal number of hidden neurons across networks with one, two, and three hidden layers
. The search boundaries for the nodes and epochs are bounded between 0 and 300
.
Fitness Evaluation: To determine the "best" position in the swarm, this methodology uses the Root Mean Square Error (RMSE) of the predictions on the validation set as the fitness function
.
Execution Details: The particle swarm is initialized with 20 particles, a maximum of 50 optimization iterations, and an inertia weight of 0.8, which regulates the search range and maintains historical velocity
.
2. Deng & Peng: Penalized Fitness Evaluation
Deng and Peng also utilize PSO, but they significantly alter both the parameters being optimized and the mathematical objective of the swarm to improve the model's ability to generalize to unseen data
.
The Search Space: In their architecture, a particle is a four-dimensional potential solution mapping to the number of neurons in the first hidden layer, the number of neurons in the second hidden layer, the dropout ratio, and the batch size
. They configure the swarm to search for a dropout rate between 0 and 1, and a batch size between 20 and 100
.
Fitness Evaluation (The Innovation): Standard PSO algorithms have poor generalization abilities and can overfit if they only optimize for minimizing prediction error
. To fix this, Deng and Peng redesign the swarm's fitness function to combine both the Mean Square Error (MSE) of the training samples and the Mean Square Weight (MSW) of the neural network
. By penalizing particles that require massive internal network weights to achieve low errors, the swarm is forced to find highly stable LSTM configurations
.
3. Ji et al.: Improved PSO (IPSO) to Prevent Local Optima
Ji et al. note that standard PSO algorithms frequently suffer from premature convergence, meaning the swarm gets trapped in a "local optimum" (a good solution, but not the absolute best) and loses the diversity needed to explore the rest of the search space
. They created an Improved PSO (IPSO) algorithm to optimize the learning rate, training epochs, and nodes of two hidden layers
.
Non-Linear Inertia Weight: Instead of keeping the inertia weight constant, Ji et al. utilize a non-linear hyperbolic tangent function
. The inertia weight starts high to force the particles to aggressively explore the global search space, and dynamically decreases as iterations continue to allow for fine-grained local searching near the end of the optimization
.
Adaptive Mutation Factor: To prevent the swarm from becoming permanently stuck, they introduce an adaptive mutation factor
. As iterations increase, the algorithm checks a randomly generated probability; if the threshold is met, the particle is forced to randomly mutate its position to escape the local trap, generating entirely new hyperparameters to test
.
The Search Space: The swarm explores learning rates within the logarithmic boundary of [0.001, 0.01], and node/epoch configurations between 1 and 200
.
4. Lanbouri & Achchab (The Exception)
It is important to note that Lanbouri and Achchab do not use Particle Swarm Optimization in their research
. Because their study is focused strictly on the impact of Technical Indicators on high-frequency, minute-by-minute intraday trading, they rely entirely on a manually configured baseline LSTM
. Their network parameters are subjectively hardcoded, utilizing exactly 5 input features connected to a single hidden layer containing exactly 10 hidden nodes
.

2.2.3 PSO Search Space
The swarm will optimize a 5-dimensional space representing the following LSTM
levers:
1. Hidden Layers: {1, 2, 3, 4}
The swarm evaluates if added depth improves directional accuracy or
merely memorizes noise. It pulls the population toward the simplest ro-
bust solution.
2. Neurons per Layer: [32, 512]
Finds the ”capacity” sweet spot to capture complex signals across 51 di-
verse tickers without redundant co-adaptation.
2
3. Learning Rate: [10−5
, 10−1]
Simultaneously tests multiple step-sizes to avoid overshooting in the high-
volatility HFT loss landscape.
4. Dropout Rate: [0.0, 0.5]
Optimizes for generalization, prioritizing particles that maintain high per-
formance on unseen validation data.
5. Lookback Window: {10, 30, 60, 120} min
Determines the optimal temporal context, discarding ”stale” data that no
longer impacts the next-minute price action.