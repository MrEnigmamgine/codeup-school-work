use employees;
-- Join Example Database
-- 1. Use the join_example_db. Select all the records from both the users and roles tables.
select *
from join_example_db.users
;
select *
from join_example_db.roles
;

-- 2. Use join, left join, and right join to combine results from the users and roles tables as we did in the lesson. 
-- Before you run each query, guess the expected number of results.
select *
from join_example_db.users
join join_example_db.roles on users.role_id = roles.id
;
-- 4

select *
from join_example_db.users
left join join_example_db.roles on users.role_id = roles.id
;
-- 6

select *
from join_example_db.users
right join join_example_db.roles on users.role_id = roles.id
;
-- 5

-- 3. Although not explicitly covered in the lesson, aggregate functions like count can be used with join queries. 
-- Use count and the appropriate join type to get a list of roles along with the number of users that has the role. Hint: You will also need to use group by in the query.
select 
	roles.name,
    count(users.id) as usr_in_role
from join_example_db.users
right join join_example_db.roles on users.role_id = roles.id
group by roles.name
;



-- Employees Database
-- 1. Use the employees database.
use employees;
-- 2. Using the example in the Associative Table Joins section as a guide, write a query that shows each department along with the name of the current manager for that department.

select 
		t2.dept_name as 'Department Name',
		CONCAT(t3.first_name, ' ', t3.last_name) as 'Department Manager'
from dept_manager as t1
join departments as t2 on t1.dept_no = t2.dept_no
join employees as t3 on t1.emp_no = t3.emp_no
where t1.to_date > now()
ORDER BY t2.dept_name asc

;

--   Department Name    | Department Manager
--  --------------------+--------------------
--   Customer Service   | Yuchang Weedman
--   Development        | Leon DasSarma
--   Finance            | Isamu Legleitner
--   Human Resources    | Karsten Sigstam
--   Marketing          | Vishwani Minakawa
--   Production         | Oscar Ghazalie
--   Quality Management | Dung Pesch
--   Research           | Hilary Kambil
--   Sales              | Hauke Zhang


-- 3. Find the name of all departments currently managed by women.
select 
		t2.dept_name as 'Department Name',
		CONCAT(t3.first_name, ' ', t3.last_name) as 'Department Manager',
        t3.gender as 'Gender'
from dept_manager as t1
join departments as t2 on t1.dept_no = t2.dept_no
join employees as t3 on t1.emp_no = t3.emp_no
where t1.to_date > now()
and t3.gender = 'F'
ORDER BY t2.dept_name asc

;

-- Department Name | Manager Name
-- ----------------+-----------------
-- Development     | Leon DasSarma
-- Finance         | Isamu Legleitner
-- Human Resources | Karsetn Sigstam
-- Research        | Hilary Kambil


-- 4. Find the current titles of employees currently working in the Customer Service department.
select * from titles;
select * from departments;

select * from dept_emp 
WHERE to_date > now()
and dept_no = 'd009';

select t2.title as 'Title',
		t1.emp_no
from dept_emp as t1
join titles as t2 on t1.emp_no = t2.emp_no
where t1.to_date > NOW()
	and t1.dept_no = 'd009'
    and title = 'Manager'
order by emp_no    
;
-- Manager	111692
-- Manager	111784
-- Manager	111877
-- Manager	111939



/************************ANSWER**************************/

select t2.title as 'Title',
		count(t1.emp_no) as 'Count'
from dept_emp as t1
join titles as t2 on t1.emp_no = t2.emp_no
where t1.to_date > NOW()
	and t1.dept_no = 'd009'
    and t2.to_date > now()
group by t2.title
order by count asc
;

-- Title              | Count
-- -------------------+------
-- Assistant Engineer |    68
-- Engineer           |   627
-- Manager            |     1
-- Senior Engineer    |  1790
-- Senior Staff       | 11268
-- Staff              |  3574
-- Technique Leader   |   241


-- 5. Find the current salary of all current managers.
select 	
		dept.dept_name as 'Department Name',
		concat(emp.first_name,' ', emp.last_name) as Name,
        sal.salary
from titles
	join salaries as sal on titles.emp_no = sal.emp_no
	join employees as emp on titles.emp_no = emp.emp_no
    join dept_emp as demp on titles.emp_no = demp.emp_no
    join departments as dept on demp.dept_no = dept.dept_no
where titles.to_date > now()
	and titles.title = 'Manager'
    and sal.to_date > now()
order by dept.dept_name
;

-- Department Name    | Name              | Salary
-- -------------------+-------------------+-------
-- Customer Service   | Yuchang Weedman   |  58745
-- Development        | Leon DasSarma     |  74510
-- Finance            | Isamu Legleitner  |  83457
-- Human Resources    | Karsten Sigstam   |  65400
-- Marketing          | Vishwani Minakawa | 106491
-- Production         | Oscar Ghazalie    |  56654
-- Quality Management | Dung Pesch        |  72876
-- Research           | Hilary Kambil     |  79393
-- Sales              | Hauke Zhang       | 101987


-- 6. Find the number of current employees in each department.
select 
		dept.dept_name,
        count(*)
from dept_emp as demp
	join departments as dept on demp.dept_no = dept.dept_no
where demp.to_date > now()
group by dept.dept_name
order by dept.dept_no asc
;

-- +---------+--------------------+---------------+
-- | dept_no | dept_name          | num_employees |
-- +---------+--------------------+---------------+
-- | d001    | Marketing          | 14842         |
-- | d002    | Finance            | 12437         |
-- | d003    | Human Resources    | 12898         |
-- | d004    | Production         | 53304         |
-- | d005    | Development        | 61386         |
-- | d006    | Quality Management | 14546         |
-- | d007    | Sales              | 37701         |
-- | d008    | Research           | 15441         |
-- | d009    | Customer Service   | 17569         |
-- +---------+--------------------+---------------+


-- 7. Which department has the highest average salary? Hint: Use current not historic information.
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
limit 1
;
-- +-----------+----------------+
-- | dept_name | average_salary |
-- +-----------+----------------+
-- | Sales     | 88852.9695     |
-- +-----------+----------------+

-- 8. Who is the highest paid employee in the Marketing department?
select
		emp.first_name,
        emp.last_name,
        sal.salary
from salaries as sal
	join dept_emp as demp on sal.emp_no = demp.emp_no
	join employees as emp on sal.emp_no = emp.emp_no
where sal.to_date > now()
	and demp.to_date > now()
    and demp.dept_no = 'd001'
order by sal.salary desc
limit 1
;
-- +------------+-----------+
-- | first_name | last_name |
-- +------------+-----------+
-- | Akemi      | Warwick   |
-- +------------+-----------+


-- 9. Which current department manager has the highest salary?

select
		dm.dept_no,
        emp.first_name,
        emp.last_name,
        sal.salary
from dept_manager as dm
	join employees as emp on dm.emp_no = emp.emp_no
    join salaries as sal on dm.emp_no = sal.emp_no
where dm.to_date > now()
	and sal.to_date > now()
order by sal.salary desc
;

-- +------------+-----------+--------+-----------+
-- | first_name | last_name | salary | dept_name |
-- +------------+-----------+--------+-----------+
-- | Vishwani   | Minakawa  | 106491 | Marketing |
-- +------------+-----------+--------+-----------+


-- 10. Determine the average salary for each department. Use all salary information and round your results.
select
		departments.dept_name,
        round(avg(sal.salary)) as average_salary
from salaries as sal
	join dept_emp as demp on sal.emp_no = demp.emp_no
	join departments on demp.dept_no = departments.dept_no
-- where sal.to_date > now()
-- 	and demp.to_date > now()
group by departments.dept_name
order by average_salary desc
;

-- +--------------------+----------------+
-- | dept_name          | average_salary | 
-- +--------------------+----------------+
-- | Sales              | 80668          | 
-- +--------------------+----------------+
-- | Marketing          | 71913          |
-- +--------------------+----------------+
-- | Finance            | 70489          |
-- +--------------------+----------------+
-- | Research           | 59665          |
-- +--------------------+----------------+
-- | Production         | 59605          |
-- +--------------------+----------------+
-- | Development        | 59479          |
-- +--------------------+----------------+
-- | Customer Service   | 58770          |
-- +--------------------+----------------+
-- | Quality Management | 57251          |
-- +--------------------+----------------+
-- | Human Resources    | 55575          |
-- +--------------------+----------------+


-- 11. Bonus Find the names of all current employees, their department name, and their current manager's name.
select
		concat(emp.first_name,' ',emp.last_name) as 'Employee Name',
        dep.dept_name as 'Department Name',
        concat(dme.first_name,' ',dme.last_name) as 'Manager Name'
from dept_emp as demp
	join employees as emp on demp.emp_no = emp.emp_no
    join departments as dep on demp.dept_no = dep.dept_no
    join dept_manager as dm on dep.dept_no = dm.dept_no
    join employees as dme on dm.emp_no = dme.emp_no
where demp.to_date > now()
	and dm.to_date > now()
order by dep.dept_name, emp.first_name
;

-- 240,124 Rows

-- Employee Name | Department Name  |  Manager Name
-- --------------|------------------|-----------------
--  Huan Lortz   | Customer Service | Yuchang Weedman

--  .....


-- 12. Bonus Who is the highest paid employee within each department.

-- The subquery I'll be using.  Returns the highest salary for each dept_no.
select
			departments.dept_no,
			max(sal.salary) as highest_salary
	from salaries as sal
		join dept_emp as demp on sal.emp_no = demp.emp_no
		join departments on demp.dept_no = departments.dept_no
	where sal.to_date > now()
		and demp.to_date > now()
	group by departments.dept_no
	order by highest_salary desc;

/*
d007	158220
d001	145128
d009	144866
*/

/***************************************************************/

select 
		concat(emp.first_name,' ',emp.last_name) as 'Name', 
        sub.highest_salary,
        dep.dept_name as 'Department Name'
from
	-- get the highest salary for each department
	(select
			departments.dept_no,
			max(sal.salary) as highest_salary
	from salaries as sal
		join dept_emp as demp on sal.emp_no = demp.emp_no
		join departments on demp.dept_no = departments.dept_no
	where sal.to_date > now()
		and demp.to_date > now()
	group by departments.dept_no
	order by highest_salary desc) as sub
-- Find the employee numbers who's salaries match the highest salary on the salaries table  
right join salaries as sal on sal.salary = sub.highest_salary 
-- We need to match that with a name, so we'll use the employee's table
join employees as emp on sal.emp_no = emp.emp_no 
-- We also want the department's name.. that lives on the departments table
join departments as dep on sub.dept_no = dep.dept_no
-- And we want to make sure that that employee is a member of that department, so tack on the dept_emp table.
join dept_emp as demp on sal.emp_no = demp.emp_no

where sal.to_date > now()  				-- make sure we aren't using old salaries
	and demp.to_date > now()			-- make sure we're using current department membership data
    and demp.dept_no = sub.dept_no		-- and finally make sure this row is for a user who is in the department we are looking for
;
/*
Khosrow Sgarro	144434	Development
Vidya Hanabata	144866	Customer Service
Tokuyasu Pesch	158220	Sales
Lunjin Swick	142395	Finance
Yinlin Flowers	141953	Human Resources
Ramachenga Soicher	130211	Research
Youjian Cronau	138273	Production
Akemi Warwick	145128	Marketing
Shin Luck	132103	Quality Management
*/