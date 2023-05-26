#include <GeneticAlgorithm/GeneticAlgorithm.h>

GeneticAlgorithm::GeneticAlgorithm(
	std::function<IIndividual* ()> createIndividual,
	size_t populationSize, size_t numberOfEpochs,
	double crossoverProbabillity, double mutationProbability) :
	m_createIndividual{ createIndividual },
	m_populationSize{ populationSize },
	m_numberOfEpochs{ numberOfEpochs },
	m_crossoverProbability{ crossoverProbabillity },
	m_mutationProbability{ mutationProbability }
{
}

void GeneticAlgorithm::Run()
{
	InitializePopulation();

	for (int index = 0; index < m_numberOfEpochs; ++index)
	{
		std::cout << std::endl << "Epoch: " << index + 1 << std::endl;

		m_fitnessValues = CalculateFitnessValues();

		Selection();
		//TournamentSelection();
		Crossover();
		//TwoPointCrossover();
		//UniformCrossoverWithCrossoverMask();
		Mutation();

		WriteWinners(index);
	}
}

IIndividual* GeneticAlgorithm::GetWinnerIndividual()
{
	double maxValue = 0.0;
	IIndividual* winner = nullptr;

	for (const auto value : m_fitnessValues)
		if (value.second > maxValue)
		{
			maxValue = value.second;
			winner = value.first;
		}

	return winner;
}

void GeneticAlgorithm::InitializePopulation()
{
	for (int index = 0; index < m_populationSize; ++index)
	{
		std::cout << "Created individual " << index + 1 << "\n";

		m_population.push_back(std::move(std::shared_ptr<IIndividual>(m_createIndividual())));
		m_workingPopulation.push_back(m_population[index]);
	}
}

std::map<IIndividual*, double> GeneticAlgorithm::CalculateFitnessValues()
{
	std::map<IIndividual*, double> fitnessValues;
	for (const auto& individual : m_workingPopulation)
	{
		double value = individual->Evaluate();
		fitnessValues[individual.get()] = value;
	}

	return fitnessValues;
}

double GeneticAlgorithm::CalculateSumOfFitnessValues()
{
	double sum{};
	for (const auto& individual : m_workingPopulation)
	{
		sum += m_fitnessValues[individual.get()];
	}

	return sum;
}

std::vector<double> GeneticAlgorithm::CalculateProbabilityOfSelection()
{
	std::vector<double> probabilityOfSelectionVector;
	double sum = CalculateSumOfFitnessValues();

	for (const auto& individual : m_workingPopulation)
	{
		probabilityOfSelectionVector.emplace_back(m_fitnessValues[individual.get()] / sum);
	}

	return probabilityOfSelectionVector;
}

std::vector<double> GeneticAlgorithm::CalcutateCumulativeProbabilityOfSelection()
{
	std::vector<double> cumulativeProbabilityOfSelectionVector;
	std::vector<double> probabilityOfSelectionVector = CalculateProbabilityOfSelection();

	for (int currentIndividualIndex = 0; currentIndividualIndex < m_populationSize; ++currentIndividualIndex)
	{
		double probability{};

		for (int index = 0; index <= currentIndividualIndex; ++index)
		{
			probability += probabilityOfSelectionVector[index];
		}

		cumulativeProbabilityOfSelectionVector.emplace_back(probability);
	}

	return cumulativeProbabilityOfSelectionVector;
}

void GeneticAlgorithm::TournamentSelection()
{
	std::vector<std::shared_ptr<IIndividual>> newPopulation;
	std::vector<size_t> selectedIndices;
	while (newPopulation.size() < m_populationSize)
	{
		selectedIndices.clear();

		for (size_t i = 0; i < TOURNAMENT_SIZE; ++i)
		{
			size_t randomIndex = RandomNumbersGenerator::GenerateIntegerNumberInRange(0, m_populationSize - 1);
			selectedIndices.push_back(randomIndex);
		}

		size_t bestIndex = selectedIndices[0];
		for (size_t i = 1; i < TOURNAMENT_SIZE; ++i)
		{
			if (m_workingPopulation[selectedIndices[i]]->Evaluate() > m_workingPopulation[bestIndex]->Evaluate())
			{
				bestIndex = selectedIndices[i];
			}
		}

		newPopulation.push_back(m_workingPopulation[bestIndex]);
	}

	m_workingPopulation = newPopulation;
}

void GeneticAlgorithm::Selection()
{
	// TODO
	std::vector<std::shared_ptr<IIndividual>>newPopulation;

	// evaluate the fitness function for each chromosome
	// sum all obtained values

	/* calculate the probability of selection for each chromosome */
	// divide the fitness value result for each chromosome by the sum of all results

	/* calculate the cumulative probability of selection */
	// the cumulative sums of the sequence {a,b,c,...}, are a, a+b, a+b+c, ....
	std::vector<double> cumulativeProbabilities = CalcutateCumulativeProbabilityOfSelection();

	/* apply the roulette wheel technique*/
	/* (spin the wheel NR times) */
	// generate m_populationSize random numbers between (0, 1] 
	std::vector<double> randomNumbers = RandomNumbersGenerator::GenerateRealNumbers(EPSILON, UPPER_BOUND, m_populationSize);

	/* (select the chromosome the wheel "landed on" according to its selection probability) */
	// for each random number generated, determine the interval (q_(i-1), q_(i)] and 
	// select in the new generation the chromosome i 
	for (size_t idx = 0; idx < randomNumbers.size();++idx)
	{
		double spinResult = randomNumbers[idx];

		if (IsGraterThan(spinResult, LOWER_BOUND) && IsLessThanOrEqualTo(spinResult, cumulativeProbabilities[0]))
		{
			newPopulation.push_back(m_workingPopulation[0]);
			continue;
		}

		for (size_t probabilityIdx = 0; probabilityIdx < m_population.size() - 1;++probabilityIdx)
		{
			if (IsGraterThan(spinResult, cumulativeProbabilities[probabilityIdx])
				&& IsLessThanOrEqualTo(spinResult, cumulativeProbabilities[probabilityIdx + 1]))
			{
				newPopulation.push_back(m_workingPopulation[probabilityIdx + 1]);
				break;
			}
		}
	}

	m_workingPopulation = newPopulation;
	// hint: make sure to handle the interval (0, q_0)
}

void GeneticAlgorithm::UniformCrossoverWithCrossoverMask()
{
	std::vector<double> randomNumbers = RandomNumbersGenerator::GenerateRealNumbers(LOWER_BOUND, UPPER_BOUND, m_populationSize);
	std::vector<std::shared_ptr<IIndividual>> selectedForCrossover;

	for (size_t idx = 0; idx < randomNumbers.size(); ++idx)
	{
		if (randomNumbers[idx] < m_crossoverProbability)
		{
			selectedForCrossover.push_back(m_workingPopulation[idx]);
		}
	}

	// Make sure the number of selected individuals for crossover is even
	if (selectedForCrossover.size() % 2 != 0)
	{
		selectedForCrossover.pop_back();
	}

	for (size_t idx = 0; idx < selectedForCrossover.size(); idx += 2)
	{
		std::shared_ptr<IIndividual> individual1 = selectedForCrossover[idx];
		std::shared_ptr<IIndividual> individual2 = selectedForCrossover[idx + 1];
		individual1->Crossover(*individual2);
	}
}

void GeneticAlgorithm::TwoPointCrossover()
{
	std::vector<double> randomNumbers = RandomNumbersGenerator::GenerateRealNumbers(LOWER_BOUND, UPPER_BOUND, m_populationSize);
	std::vector<std::shared_ptr<IIndividual>> selectedForCrossover;

	for (size_t idx = 0; idx < randomNumbers.size(); ++idx)
	{
		if (randomNumbers[idx] < m_crossoverProbability)
		{
			selectedForCrossover.push_back(m_workingPopulation[idx]);
		}
	}

	// Make sure the number of selected individuals for crossover is even
	if (selectedForCrossover.size() % 2 != 0)
	{
		selectedForCrossover.pop_back();
	}

	for (size_t idx = 0; idx < selectedForCrossover.size(); idx += 2)
	{
		std::shared_ptr<IIndividual> individual1 = selectedForCrossover[idx];
		std::shared_ptr<IIndividual> individual2 = selectedForCrossover[idx + 1];
		individual1->TwoPointCrossover(*individual2);
	}
}

void GeneticAlgorithm::Crossover()
{
	// TODO

	// for each individual in the population, generate a random number between (0, 1]
	// apply crossover only for those individuals for which the random value is lower than the crossover probability
	std::vector<double> randomNumbers = RandomNumbersGenerator::GenerateRealNumbers(LOWER_BOUND, UPPER_BOUND, m_populationSize);
	std::vector<std::shared_ptr<IIndividual>> selectedForCrossover;

	for (size_t idx = 0;idx < randomNumbers.size();++idx)
	{
		if (randomNumbers[idx] < m_crossoverProbability)
		{
			selectedForCrossover.push_back(m_workingPopulation[idx]);
		}
	}
	// note that crossover is a binary operation

	if (selectedForCrossover.size() % 2 != 0)
	{
		selectedForCrossover.pop_back();
	}

	for (size_t idx = 0;idx < selectedForCrossover.size();idx += 2)
	{
		selectedForCrossover[idx]->Crossover(*selectedForCrossover[idx + 1]);
	}
}

void GeneticAlgorithm::Mutation()
{
	// TODO

	// for every individual in the population, we will apply our custom 
	// mutation implementation
	for (size_t idx = 0;idx < m_populationSize;++idx)
	{
		m_workingPopulation[idx]->Mutation(m_mutationProbability);
	}
}

bool GeneticAlgorithm::IsGraterThan(double value, double lowerBound) const
{
	return value > lowerBound;
}

bool GeneticAlgorithm::IsLessThanOrEqualTo(double value, double upperBound) const
{
	return value <= upperBound;
}

void GeneticAlgorithm::WriteWinners(int epoch)
{
	IIndividual* winner = GetWinnerIndividual();

	if (epoch == 0)
	{
		IOIndividualManager::WriteIndividualValueInFile(epoch + 1, m_fitnessValues[winner], false);
	}
	else
	{
		if (epoch == m_numberOfEpochs - 1)
		{
			IOIndividualManager::WriteIndividualDetailsInFile(winner);
		}
		IOIndividualManager::WriteIndividualValueInFile(epoch + 1, m_fitnessValues[winner], true);
	}
}
