-- 1. Create a file named where_exercises.sql. Make sure to use the employees database.
use employees;

-- 2. Find all current or previous employees with first names 'Irena', 'Vidya', or 'Maya' using IN. Enter a comment with the number of records returned.
select * from employees where first_name in ('Irena','Vidya','Maya'); 
-- 709 rows returned

-- 3. Find all current or previous employees with first names 'Irena', 'Vidya', or 'Maya', as in Q2, but use OR instead of IN. Enter a comment with the number of records returned. Does it match number of rows from Q2?
select * from employees where 
first_name = 'Irena'
OR first_name = 'Vidya'
OR first_name = 'Maya';
-- 709 rows returned, Yes

-- 4. Find all current or previous employees with first names 'Irena', 'Vidya', or 'Maya', using OR, and who is male. Enter a comment with the number of records returned.
select * from employees where 
(first_name = 'Irena'
OR first_name = 'Vidya'
OR first_name = 'Maya')
AND gender = 'M';
-- 441 rows returned

-- 5. Find all current or previous employees whose last name starts with 'E'. Enter a comment with the number of employees whose last name starts with E.
select * from employees where 
last_name LIKE 'E%';
-- LIMIT 10000000
-- 7330 rows returned

-- 6. Find all current or previous employees whose last name starts or ends with 'E'. 
-- Enter a comment with the number of employees whose last name starts or ends with E. How many employees have a last name that ends with E, but does not start with E?
select * from employees where 
last_name LIKE 'E%'
OR last_name LIKE '%E';
-- 30723 rows returned

-- How many employees have a last name that ends with E, but does not start with E?
select * from employees where 
last_name NOT LIKE 'E%'
AND last_name LIKE '%E';
-- 23393 rows returned

-- 7. Find all current or previous employees employees whose last name starts and ends with 'E'. 
-- Enter a comment with the number of employees whose last name starts and ends with E. 
select * from employees where 
last_name LIKE 'E%'
AND last_name LIKE '%E';
-- 899 rows returned

-- How many employees' last names end with E, regardless of whether they start with E?
select * from employees where 
last_name LIKE '%E';
-- 24292 rows returned

-- 8. Find all current or previous employees hired in the 90s. Enter a comment with the number of employees returned.
select * from employees where 
hire_date between '1990-01-01' AND '1999-12-31'
order by birth_date asc;
-- 135214 rows returned

-- 9. Find all current or previous employees born on Christmas. Enter a comment with the number of employees returned.
select * from employees where 
birth_date like '%-12-25';
-- 842 rows returned

-- 10. Find all current or previous employees hired in the 90s and born on Christmas. Enter a comment with the number of employees returned.
select * from employees where 
hire_date between '1990-01-01' AND '1999-12-31'
AND birth_date like '%-12-25';
-- 362 rows returned
 
-- 11. Find all current or previous employees with a 'q' in their last name. Enter a comment with the number of records returned.
select * from employees where 
last_name LIKE '%q%';
-- 1873 rows returned

-- 12. Find all current or previous employees with a 'q' in their last name but not 'qu'. How many employees are found?
select * from employees where 
last_name LIKE '%q%'
AND last_name NOT LIKE '%qu%';
-- 547 rows returned