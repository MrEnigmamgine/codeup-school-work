-- Create a new file named order_by_exercises.sql and copy in the contents of your exercise from the previous lesson.
use employees;
select * from employees where 
first_name = 'Irena'
OR first_name = 'Vidya'
OR first_name = 'Maya';

-- Find all employees with first names 'Irena', 'Vidya', or 'Maya', and order your results returned by first name. 
-- In your comments, answer: What was the first and last name in the first row of the results? 
-- What was the first and last name of the last person in the table?
select * from employees 
where first_name in ('Irena','Vidya','Maya')
order by first_name;
-- Irena Reutnauer
-- Vidya Simmen

-- Find all employees with first names 'Irena', 'Vidya', or 'Maya', and order your results returned by first name and then last name. 
-- In your comments, answer: What was the first and last name in the first row of the results? 
-- What was the first and last name of the last person in the table?
select * from employees 
where first_name in ('Irena','Vidya','Maya')
order by first_name, last_name;
-- Irena Acton
-- Vidya Zweizig

-- Find all employees with first names 'Irena', 'Vidya', or 'Maya', and order your results returned by last name and then first name. 
-- In your comments, answer: What was the first and last name in the first row of the results? 
-- What was the first and last name of the last person in the table?
select * from employees 
where first_name in ('Irena','Vidya','Maya')
order by last_name, first_name;
-- Irena Acton
-- Maya Zyda


-- Write a query to to find all employees whose last name starts and ends with 'E'. Sort the results by their employee number. 
-- Enter a comment with the number of employees returned, the first employee number and their first and last name, and the last employee number with their first and last name.
select * from employees where 
last_name LIKE 'E%'
AND last_name LIKE '%E'
ORDER BY emp_no;
-- 899 rows returned
-- 10021 Ramzi Erde
-- 499648 Tadahiro Erde

-- Write a query to to find all employees whose last name starts and ends with 'E'. Sort the results by their hire date, so that the newest employees are listed first. 
-- Enter a comment with the number of employees returned, the name of the newest employee, and the name of the oldest employee.
select * from employees where 
last_name LIKE 'E%'
AND last_name LIKE '%E'
ORDER BY hire_date DESC;
-- 899 rows returned
-- Teiji Eldridge
-- Sergi Erde


-- Find all employees hired in the 90s and born on Christmas. Sort the results so that the oldest employee who was hired last is the first result. 
-- Enter a comment with the number of employees returned, the name of the oldest employee who was hired last, and the name of the youngest employee who was hired first.
select * from employees where 
hire_date between '1990-01-01' AND '1999-12-31'
AND birth_date like '%-12-25'
ORDER BY birth_date ASC, hire_date DESC;
-- 362 rows returned
-- Khun Bernini
-- Douadi Pettis

/************************************************************************************************************/
/***************************************Functions Exercises**************************************************/
/************************************************************************************************************/

-- Write a query to to find all employees whose last name starts and ends with 'E'. Use concat() to combine their first and last name together as a single column named full_name.
SELECT *,
CONCAT(first_name,last_name) as full_name
FROM employees
WHERE last_name LIKE 'e%e';

-- Convert the names produced in your last query to all uppercase.
SELECT *,
UPPER(CONCAT(first_name,last_name)) as full_name
FROM employees
WHERE last_name LIKE 'e%e';
-- Find all employees hired in the 90s and born on Christmas. Use datediff() function to find how many days they have been working at the company 
-- (Hint: You will also need to use NOW() or CURDATE()),
select * ,
DATEDIFF(NOW(),hire_date) as tenure
from employees where 
hire_date between '1990-01-01' AND '1999-12-31'
AND birth_date like '%-12-25';

-- Find the smallest and largest current salary from the salaries table.
SELECT min(salary), max(salary)
FROM employees.salaries
WHERE to_date = '9999-01-01';

-- Use your knowledge of built in SQL functions to generate a username for all of the employees. 
-- A username should be all lowercase, and consist of the first character of the employees first name, the first 4 characters of the employees last name, 
-- an underscore, the month the employee was born, and the last two digits of the year that they were born. Below is an example of what the first 10 rows will look like:

-- +------------+------------+-----------+------------+
-- | username   | first_name | last_name | birth_date |
-- +------------+------------+-----------+------------+
-- | gface_0953 | Georgi     | Facello   | 1953-09-02 |
-- | bsimm_0664 | Bezalel    | Simmel    | 1964-06-02 |
-- | pbamf_1259 | Parto      | Bamford   | 1959-12-03 |
-- | ckobl_0554 | Chirstian  | Koblick   | 1954-05-01 |
-- | kmali_0155 | Kyoichi    | Maliniak  | 1955-01-21 |
-- | apreu_0453 | Anneke     | Preusig   | 1953-04-20 |
-- | tziel_0557 | Tzvetan    | Zielinski | 1957-05-23 |
-- | skall_0258 | Saniya     | Kalloufi  | 1958-02-19 |
-- | speac_0452 | Sumant     | Peac      | 1952-04-19 |
-- | dpive_0663 | Duangkaew  | Piveteau  | 1963-06-01 |
-- +------------+------------+-----------+------------+
-- 10 rows in set (0.05 sec)

SELECT 	first_name,
		last_name,
		birth_date,
        LOWER(CONCAT(substr(first_name, 1,1),substr(last_name,1,4),'_',substr(birth_date,6,2),substr(YEAR(birth_date),3,2))) as username
        -- substr(YEAR(birth_date),3,2)
FROM employees.employees
LIMIT 10;