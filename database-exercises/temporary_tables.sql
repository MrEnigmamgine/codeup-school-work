use employees;

/**************** Refresh *****************/
drop table if exists kalpana_1811.employees_with_departments;

-- 1. create a temporary table called employees_with_departments that contains first_name, last_name, and dept_name for employees currently with that department

CREATE TEMPORARY TABLE kalpana_1811.employees_with_departments AS
	SELECT emp_no, first_name, last_name, dept_no, dept_name
	FROM employees
	JOIN dept_emp USING(emp_no)
	JOIN departments USING(dept_no)
;

-- 1.A Add a column named full_name to this table. It should be a VARCHAR whose length is the sum of the lengths of the first name and last name columns
describe kalpana_1811.employees_with_departments;
/*emp_no	int
first_name	varchar(14)
last_name	varchar(16)
dept_no	char(4)
dept_name	varchar(40)*/
alter table kalpana_1811.employees_with_departments add full_name varchar(30);

-- 1.B Update the table so that full name column contains the correct data
update kalpana_1811.employees_with_departments
	set full_name = concat(first_name,' ',last_name)
;
select * from kalpana_1811.employees_with_departments limit 100;

-- 1.C Remove the first_name and last_name columns from the table.
alter table kalpana_1811.employees_with_departments drop column first_name;
alter table kalpana_1811.employees_with_departments drop column last_name;
select * from kalpana_1811.employees_with_departments limit 100;

-- 1.D What is another way you could have ended up with this same table?
-- Include the concatonation in the original query and don't select first_name or last_name.

CREATE TEMPORARY TABLE kalpana_1811.employees_with_departments AS
	SELECT emp_no, 
    concat(first_name,' ', last_name)
    dept_no, dept_name
	FROM employees
	JOIN dept_emp USING(emp_no)
	JOIN departments USING(dept_no)
;


-- 2. Create a temporary table based on the payment table from the sakila database.
-- Write the SQL necessary to transform the amount column such that it is stored as an integer representing the number of cents of the payment. 
-- For example, 1.99 should become 199.

use sakila;
select * from payment limit 100;
/**************** Refresh *****************/
drop table if exists kalpana_1811.sakila_cents;
CREATE TEMPORARY TABLE kalpana_1811.sakila_cents AS
	select *,
		amount * 100 div 1 as cents 
        /* Integer division discards from the division result any fractional part to the right of the decimal point. 
        This results in an INT*/
	from payment
;
select * from kalpana_1811.sakila_cents;
describe kalpana_1811.sakila_cents;

	/* 3
	Find out how the current average pay in each department compares to the overall current pay for everyone at the company. 
	In order to make the comparison easier, you should use the Z-score for salaries. In terms of salary, what is the best department right now to work for? The worst?

	Hint Consider that the following code will produce the z score for current salaries.
		-- Returns the historic z-scores for each salary
		-- Notice that there are 2 separate scalar subqueries involved
		SELECT salary,
			(salary - (SELECT AVG(salary) FROM salaries))
			/
			(SELECT stddev(salary) FROM salaries) AS zscore
		FROM salaries;
        */

use employees;

/**************** Refresh *****************/
drop table if exists kalpana_1811.avg_salary_by_department_current;
create temporary table kalpana_1811.avg_salary_by_department_current as
select
		departments.dept_name,
        avg(sal.salary) as average_salary
from salaries as sal
	join dept_emp as demp on sal.emp_no = demp.emp_no
	join departments on demp.dept_no = departments.dept_no
where sal.to_date > now()
 	and demp.to_date > now()
group by departments.dept_name
order by average_salary desc
;
select * from kalpana_1811.avg_salary_by_department_current;


select *,
	/* (Department average - Current company wide average) / Current company wide stddev */
	(average_salary - (SELECT AVG(salary) FROM salaries where to_date > now()))
	/
	(SELECT stddev(salary) FROM salaries where to_date > now()) AS zscore
from kalpana_1811.avg_salary_by_department_current
order by zscore desc
;

-- The best department to work for is Sales with a zscore of 0.97
-- The worst is Human Resources with a zscore of -0.46

/************* BONUS *****************/
-- To your work with current salary zscores, determine 
-- - the overall historic average departement average salary, 

/**************** Refresh *****************/
drop table if exists kalpana_1811.avg_salary_by_department_historical;
create temporary table kalpana_1811.avg_salary_by_department_historical as
select
		departments.dept_name,
        avg(sal.salary) as average_salary
from salaries as sal
	join dept_emp as demp on sal.emp_no = demp.emp_no
	join departments on demp.dept_no = departments.dept_no
-- where sal.to_date > now()
-- 	and demp.to_date > now()
group by departments.dept_name
order by average_salary desc
;

select * from kalpana_1811.avg_salary_by_department_historical;

-- - the historic overall average, 
select avg(salary) from salaries; -- historical avg salary (63810.7448)

-- - and the historic zscores for salary. 
select *,
	(average_salary - (SELECT AVG(salary) FROM salaries ))
	/
	(SELECT stddev(salary) FROM salaries ) AS zscore
from kalpana_1811.avg_salary_by_department_historical
order by zscore desc
;

-- Do the zscores for current department average salaries tell a similar or a different story than the historic department salary zscores?

select 
	*
from (
	select dept_name, average_salary as cur_average_salary,
	(average_salary - (SELECT AVG(salary) FROM salaries where to_date > now()))
	/
	(SELECT stddev(salary) FROM salaries where to_date > now()) AS cur_zscore
	from kalpana_1811.avg_salary_by_department_current
    ) as cur
join (
	select *,
	(average_salary - (SELECT AVG(salary) FROM salaries ))
	/
	(SELECT stddev(salary) FROM salaries ) AS hist_zscore
	from kalpana_1811.avg_salary_by_department_historical
    ) as hist using (dept_name)
;



/***************** Scratchpad ****************/
select stddev(average_salary) from kalpana_1811.avg_salary_by_department_historical;