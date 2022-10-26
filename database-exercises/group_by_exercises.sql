-- describe employees.titles;
-- -- +-----------+-------------+------+-----+---------+-------+
-- -- | Field     | Type        | Null | Key | Default | Extra |
-- -- +-----------+-------------+------+-----+---------+-------+
-- -- | emp_no    | int         | NO   | PRI | NULL    |       |
-- -- | title     | varchar(50) | NO   | PRI | NULL    |       |
-- -- | from_date | date        | NO   | PRI | NULL    |       |
-- -- | to_date   | date        | YES  |     | NULL    |       |
-- -- +-----------+-------------+------+-----+---------+-------+

-- In your script, use DISTINCT to find the unique titles in the titles table. How many unique titles have there ever been? Answer that in a comment in your SQL file.
SELECT distinct title
FROM employees.titles;
-- 7 unique titles

-- Write a query to to find a list of all unique last names of all employees that start and end with 'E' using GROUP BY.
SELECT last_name
from employees.employees
where last_name like 'e%e'
group by last_name;

-- Write a query to to find all unique combinations of first and last names of all employees whose last names start and end with 'E'.
SELECT last_name,
		first_name
from employees.employees
where last_name like 'e%e'
group by last_name, first_name
;
-- Write a query to find the unique last names with a 'q' but not 'qu'. Include those names in a comment in your sql code.
select distinct last_name
from employees where 
last_name LIKE '%q%'
AND last_name NOT LIKE '%qu%'
;
-- Chleq
-- Lindqvist
-- Qiwen

-- Add a COUNT() to your results (the query above) to find the number of employees with the same last name.
select count(last_name), last_name
from employees where 
last_name LIKE '%q%'
AND last_name NOT LIKE '%qu%'
group by last_name
;
-- 189	Chleq
-- 190	Lindqvist
-- 168	Qiwen

-- Find all all employees with first names 'Irena', 'Vidya', or 'Maya'. Use COUNT(*) and GROUP BY to find the number of employees for each gender with those names.
SELECT 
    gender, 
    count(gender)
FROM
    employees
WHERE
    first_name IN ('Irena', 'Vidya', 'Maya')
group by gender
;
-- Using your query that generates a username for all of the employees, generate a count employees for each unique username. Are there any duplicate usernames? 
-- BONUS: How many duplicate usernames are there?

SELECT 
    LOWER(CONCAT(SUBSTR(first_name, 1, 1),
                    SUBSTR(last_name, 1, 4),
                    '_',
                    SUBSTR(birth_date, 6, 2),
                    SUBSTR(YEAR(birth_date), 3, 2))) AS username,
	count(*) as usernameCount
FROM
    employees.employees
group by username
having usernameCount > 1
;
-- BONUS: How many duplicate usernames are there?
-- 27403
SELECT sum(g.usernameCount) 
FROM
	(SELECT 
		LOWER(CONCAT(SUBSTR(first_name, 1, 1),
						SUBSTR(last_name, 1, 4),
						'_',
						SUBSTR(birth_date, 6, 2),
						SUBSTR(YEAR(birth_date), 3, 2))) AS username,
		count(*) as usernameCount
	FROM
		employees.employees
	group by username
	having usernameCount > 1
	) as g;


-- Bonus: More practice with aggregate functions:

-- Determine the historic average salary for each employee. When you hear, read, or think "for each" with regard to SQL, you'll probably be grouping by that exact column.
SELECT 
    emp_no, 
    AVG(salary)
FROM
    employees.salaries
GROUP BY emp_no
;
-- Using the dept_emp table, count how many current employees work in each department. The query result should show 9 rows, one for each department and the employee count.
SELECT
	dept_no,
    count(*)
FROM
	employees.dept_emp
WHERE to_date > NOW()
group by dept_no
;

-- Determine how many different salaries each employee has had. This includes both historic and current.
-- Find the maximum salary for each employee.
-- Find the minimum salary for each employee.
SELECT 
    emp_no, 
    count(salary),
    max(salary),
    min(salary)
FROM
    employees.salaries
GROUP BY emp_no
;


-- Find the standard deviation of salaries for each employee.


-- Now find the max salary for each employee where that max salary is greater than $150,000.
SELECT g.emp_no, g.sMax
FROM
	(SELECT 
		emp_no, 
		count(salary) as sCount,
		max(salary) as sMax,
		min(salary) as sMin
	FROM
		employees.salaries
	GROUP BY emp_no) as g
WHERE
	sMax > 150000;
-- Find the average salary for each employee where that average salary is between $80k and $90k.

SELECT g.emp_no, g.sAvg
	FROM
		(SELECT 
			emp_no, 
			count(salary) as sCount,
			max(salary) as sMax,
			min(salary) as sMin,
			avg(salary) as sAvg
		FROM
			employees.salaries
		GROUP BY emp_no) as g
	WHERE
		sAvg between 80000 and 90000
;