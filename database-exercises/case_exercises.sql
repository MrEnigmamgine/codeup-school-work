use employees;

-- 1. Write a query that returns all employees, their department number, their start date, their end date, 
-- and a new column 'is_current_employee' that is a 1 if the employee is still with the company and 0 if not.
/******************* WIP *****************/
SELECT
	distinct (emp_no),
    dept_no,
    hire_date,
    to_date,
    if(to_date > now(), TRUE, FALSE) as is_current_employee
FROM
	employees
    left join dept_emp as emp using (emp_no)
WHERE
	to_date > now()
    
UNION SELECT
	distinct (emp_no),
    dept_no,
    hire_date,
    to_date,
    if(to_date > now(), TRUE, FALSE) as is_current_employee
FROM
	employees
    left join dept_emp as emp using (emp_no)
WHERE
	to_date < now()
    and emp_no not in (
						select emp_no 
                        from employees
                        join dept_emp as emp using (emp_no)
                        where to_date > now()
                        )
order by emp_no asc
;
/* This will return employees more than once.  I'm not sure how to easily go about filtering it so if there's a newer record to drop the older record. */

/*************** Q1 attempt 2 ************/
select max(to_date), emp_no from dept_emp group by emp_no
; -- latest to_date per employee

select 
	de.emp_no,
    dept_no,
    hire_date,
    de.to_date,
    if(de.to_date > now(), TRUE, FALSE) as is_current_employee
from dept_emp as de
	join (select max(to_date) as to_date, emp_no from dept_emp group by emp_no) as sub
		on de.emp_no = sub.emp_no
        and de.to_date = sub.to_date
	join employees emp on de.emp_no = emp.emp_no
        
;

-- 2. Write a query that returns all employee names (previous and current), 
-- and a new column 'alpha_group' that returns 'A-H', 'I-Q', or 'R-Z' depending on the first letter of their last name.

select first_name, last_name,
	case 
		when substr(last_name, 1, 1) between 'A' and 'H' then 'A-H'
        when substr(last_name, 1, 1) between 'I' and 'Q' then 'I-Q'
        when substr(last_name, 1, 1) between 'R' and 'Z' then 'R-Z'
	end as alpha_group
from employees
;

-- 3. How many employees (current or previous) were born in each decade?
SELECT
	concat(floor(year(birth_date) / 10) * 10, 's') as decade_born,
    count(*)
from employees
group by decade_born
;

/************************ scratchpad ********************/
select max(birth_date), min(birth_date) from employees;

-- 4. What is the current average salary for each of the following department groups: R&D, Sales & Marketing, Prod & QM, Finance & HR, Customer Service?
SELECT 
   CASE
       WHEN dept_name IN ('research', 'development') THEN 'R&D'
       WHEN dept_name IN ('sales', 'marketing') THEN 'Sales & Marketing'
       WHEN dept_name IN ('Production', 'Quality Management') THEN 'Prod & QM'
       WHEN dept_name IN ('Finance', 'Human Resources') THEN 'Finance & HR'
       ELSE dept_name
   END AS dept_group,
   avg(salary)
FROM departments
	join dept_emp as demp using (dept_no)
    join salaries as sal using (emp_no)
where
	demp.to_date > now()
	and sal.to_date > now()
group by dept_group
;
