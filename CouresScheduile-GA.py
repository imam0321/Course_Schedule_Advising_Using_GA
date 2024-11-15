import pandas as pd
import random
import matplotlib.pyplot as plt

# Load CSV files
preregistration = pd.read_csv('/content/HasedPreregistartion.csv')
routine = pd.read_csv('/content/Routine.csv')

# Constants
MAX_DAYS = 4
MAX_BREAK = 2
MAX_STUDENTS_PER_SECTION = 35
MIN_STUDENTS_PER_SECTION = 20

# Parse time function with day mapping
def parse_time(row):
    start = row['Start']
    end = row['End']

    if len(start) > 5:
        day_code = start[0:2]
        start_time = start[2:]
        end_time = end[2:]
    else:
        day_code = start[0]
        start_time = start[1:]
        end_time = end[1:]

    day_mapping = {
        'S': 'Sunday',
        'M': 'Monday',
        'T': 'Tuesday',
        'W': 'Wednesday',
        'H': 'Thursday',
        'F': 'Friday',
        'A': 'Saturday',
        'ST': ['Sunday', 'Tuesday'],
        'MW': ['Monday', 'Wednesday'],
        'HA': ['Thursday', 'Saturday']
    }

    day = day_mapping.get(day_code, day_code)
    return (day, start_time, end_time)

# Apply the parse_time function to the Routine DataFrame
routine['Parsed'] = routine.apply(parse_time, axis=1)

# Create a random chromosome (student schedule assignment)
def create_chromosome():
    chromosome = []
    for _, student in preregistration.iterrows():
        student_courses = student['Preregistered Courses'].split(';')
        assigned_courses = []
        for course in student_courses:
            possible_timeslots = routine['Parsed'].tolist()
            assigned_timeslot = random.choice(possible_timeslots)
            assigned_courses.append((course, assigned_timeslot))
        chromosome.append(assigned_courses)
    return chromosome

# Fitness function with minimized penalties for constraints
def fitness(chromosome):
    total_days_penalty = 0
    total_conflict_penalty = 0
    total_class_size_penalty = 0
    total_class_sizes = {}

    for student_courses in chromosome:
        student_days = set()
        for course, timeslot in student_courses:
            day, start_time, end_time = timeslot
            if isinstance(day, list):  # Multiple days like MW
                student_days.update(day)
            else:
                student_days.add(day)

        # Penalty if a student exceeds MAX_DAYS
        if len(student_days) > MAX_DAYS:
            total_days_penalty += (len(student_days) - MAX_DAYS) * 5

        # Check for time conflicts
        for i, (course1, timeslot1) in enumerate(student_courses):
            for j, (course2, timeslot2) in enumerate(student_courses):
                if i < j:  # Avoid redundant checks
                    day1, start_time1, end_time1 = timeslot1
                    day2, start_time2, end_time2 = timeslot2

                    # Only check if the courses are on the same day(s)
                    if set(day1) & set(day2):  # Intersection of days
                        # Check for time overlap
                        if (start_time1 < end_time2 and start_time2 < end_time1):
                            total_conflict_penalty += 50  # Higher penalty weight

        # Section size penalty
        for course, timeslot in student_courses:
            # Convert timeslot to a hashable type
            day, start_time, end_time = timeslot
            if isinstance(day, list):
                day = tuple(day)  # Convert list of days to a tuple

            hashable_timeslot = (day, start_time, end_time)

            if (course, hashable_timeslot) not in total_class_sizes:
                total_class_sizes[(course, hashable_timeslot)] = 0
            total_class_sizes[(course, hashable_timeslot)] += 1

    # Apply penalties for section size
    for section, size in total_class_sizes.items():
        if size < MIN_STUDENTS_PER_SECTION:
            total_class_size_penalty += (MIN_STUDENTS_PER_SECTION - size) * 10
        elif size > MAX_STUDENTS_PER_SECTION:
            total_class_size_penalty += (size - MAX_STUDENTS_PER_SECTION) * 10

    # Combine penalties for the final score
    fitness_score = total_days_penalty + total_conflict_penalty + total_class_size_penalty
    return fitness_score, total_class_sizes

# Crossover function (combine two chromosomes to generate a new one)
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]

# Adaptive mutation function
def mutation(chromosome, generation, max_generations):
    mutation_probability = 0.5 * (1 - generation / max_generations)
    if random.random() < mutation_probability:
        student_idx = random.randint(0, len(chromosome) - 1)
        course_idx = random.randint(0, len(chromosome[student_idx]) - 1)
        new_timeslot = random.choice(routine['Parsed'].tolist())
        chromosome[student_idx][course_idx] = (chromosome[student_idx][course_idx][0], new_timeslot)
    return chromosome

# Repair function to resolve conflicts
def repair_conflicts(student_courses):
    for i, item1 in enumerate(student_courses):
        # Ensure item is a valid tuple
        if isinstance(item1, tuple) and len(item1) == 2:
            course1, timeslot1 = item1
            for j, item2 in enumerate(student_courses):
                if i < j and isinstance(item2, tuple) and len(item2) == 2:
                    course2, timeslot2 = item2
                    day1, start_time1, end_time1 = timeslot1
                    day2, start_time2, end_time2 = timeslot2

                    # Check for conflicts
                    if set(day1) & set(day2) and (start_time1 < end_time2 and start_time2 < end_time1):
                        # Resolve conflict by assigning a new timeslot
                        student_courses[j] = (course2, random.choice(routine['Parsed'].tolist()))
    return student_courses

# Selection function with elitism
def selection(population, num_parents):
    sorted_population = sorted(population, key=lambda x: fitness(x)[0])
    return sorted_population[:num_parents]

# Genetic Algorithm function
def genetic_algorithm():
    population_size = 350
    num_generations = 150
    num_parents = 50
    max_generations = num_generations

    population = [create_chromosome() for _ in range(population_size)]
    fitness_history = []

    for generation in range(num_generations):
        parents = selection(population, num_parents)
        next_generation = parents[:5]  # Keep top 5 as elite

        while len(next_generation) < population_size:
            parent1, parent2 = random.choice(parents), random.choice(parents)
            child = crossover(parent1, parent2)
            child = mutation(child, generation, max_generations)
            child = [repair_conflicts(courses) for courses in child]  # Repair conflicts after mutation
            next_generation.append(child)

        population = next_generation

        best_solution = min(population, key=lambda x: fitness(x)[0])
        best_fitness, class_sizes = fitness(best_solution)
        fitness_history.append(best_fitness)

        print(f"Generation {generation}: Best fitness = {best_fitness}")
        print("Class Sizes for this generation:")
        for section, size in class_sizes.items():
            print(f"  {section}: {size} students")
        print("\n")

    return min(population, key=lambda x: fitness(x)[0]), fitness_history

# Print function for student schedule
def print_student_schedule(best_solution):
    for student_idx, student_courses in enumerate(best_solution):
        print(f"Student {student_idx + 1} Schedule:")
        for course, timeslot in student_courses:
            days, start_time, end_time = timeslot
            days_str = ', '.join(days) if isinstance(days, list) else days
            print(f"  {course} on {days_str} from {start_time} to {end_time}")
        print("\n")

# Print a table for course section sizes and total students per section
def print_class_sizes(class_sizes):
    # Prepare data for the table
    class_data = []
    for section, size in class_sizes.items():
        course, timeslot = section
        day_str = ', '.join(timeslot[0]) if isinstance(timeslot[0], list) else timeslot[0]
        start_time = timeslot[1]
        end_time = timeslot[2]
        class_data.append({
            'Course': course,
            'Section': f"{day_str} from {start_time} to {end_time}",
            'Total Students': size
        })

    # Create a DataFrame and print it
    class_df = pd.DataFrame(class_data)
    print(class_df.to_string(index=False))

# Run the Genetic Algorithm
best_solution, fitness_history = genetic_algorithm()

# Print the best solution found
print("Best solution found:")
print_student_schedule(best_solution)

# Print the class sizes for the best solution
best_fitness, class_sizes = fitness(best_solution)
print_class_sizes(class_sizes)

# Plot the fitness improvement over generations
plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness Improvement over Generations')
plt.show()
