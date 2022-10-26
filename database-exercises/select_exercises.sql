-- 2. Use the albums_db database
USE albums_db;

-- 3. Explore the structure of the albums table.
describe albums;
/*+--------------+--------------+------+-----+---------+----------------+
| Field        | Type         | Null | Key | Default | Extra          |
+--------------+--------------+------+-----+---------+----------------+
| id           | int unsigned | NO   | PRI | NULL    | auto_increment |
| artist       | varchar(240) | YES  |     | NULL    |                |
| name         | varchar(240) | NO   |     | NULL    |                |
| release_date | int          | YES  |     | NULL    |                |
| sales        | float        | YES  |     | NULL    |                |
| genre        | varchar(240) | YES  |     | NULL    |                |
+--------------+--------------+------+-----+---------+----------------+*/

-- 3.a. How many rows are in the albums table?
select id from albums;
-- 31 -- 31 row(s) returned

-- 3.b. How many unique artist names are in the albums table?
SELECT DISTINCT artist FROM albums;
-- 23 -- 23 row(s) returned

-- 3.c. What is the primary key for the albums table?
-- id

-- 3.d. What is the oldest release date for any album in the albums table? What is the most recent release date?
SELECT
MIN(release_date),
MAX(release_date)
FROM albums;
/*+-------------------+-------------------+
| MIN(release_date) | MAX(release_date) |
+-------------------+-------------------+
|              1967 |              2011 |
+-------------------+-------------------+*/
-- 1967 , 2011

-- 4. Write queries to find the following information:
-- 4.a. The name of all albums by Pink Floyd
SELECT name FROM albums_db.albums WHERE artist = 'Pink Floyd';
-- +---------------------------+
-- | name                      |
-- +---------------------------+
-- | The Dark Side of the Moon |
-- | The Wall                  |
-- +---------------------------+

-- 4.b.  The year Sgt. Pepper's Lonely Hearts Club Band was released
SELECT release_date, name, artist FROM albums_db.albums WHERE name = 'Sgt. Pepper''s Lonely Hearts Club Band';
-- +--------------+---------------------------------------+-------------+
-- | release_date | name                                  | artist      |
-- +--------------+---------------------------------------+-------------+
-- |         1967 | Sgt. Pepper's Lonely Hearts Club Band | The Beatles |
-- +--------------+---------------------------------------+-------------+

-- 4.c. The genre for the album Nevermind
SELECT genre FROM albums_db.albums WHERE name = 'Nevermind';
-- 'Grunge, Alternative rock'

-- 4.d. Which albums were released in the 1990s
SELECT name, release_date FROM albums_db.albums WHERE release_date BETWEEN 1990 AND 1999; 
-- +----------------------------------------+-----------------------------------+--------------+
-- | name                                   | artist                            | release_date |
-- +----------------------------------------+-----------------------------------+--------------+
-- | The Bodyguard                          | Whitney Houston / Various artists |         1992 |
-- | Jagged Little Pill                     | Alanis Morissette                 |         1995 |
-- | Come On Over                           | Shania Twain                      |         1997 |
-- | Falling into You                       | Celine Dion                       |         1996 |
-- | Let's Talk About Love                  | Celine Dion                       |         1997 |
-- | Dangerous                              | Michael Jackson                   |         1991 |
-- | The Immaculate Collection              | Madonna                           |         1990 |
-- | Titanic: Music from the Motion Picture | James Horner                      |         1997 |
-- | Metallica                              | Metallica                         |         1991 |
-- | Nevermind                              | Nirvana                           |         1991 |
-- | Supernatural                           | Santana                           |         1999 |
-- +----------------------------------------+-----------------------------------+--------------+

-- 4.e. Which albums had less than 20 million certified sales
SELECT name, artist, sales FROM albums_db.albums WHERE sales > 20;

-- 4.f. All the albums with a genre of "Rock". Why do these query results not include albums with a genre of "Hard rock" or "Progressive rock"?
-- This query looks for albums with the string "Rock" stored in the genre field.  "Hard rock" != "Rock". 
-- If we wanted to include those we would use the LIKE operator.
-- SELECT name, artist, genre FROM albums_db.albums WHERE genre LIKE '%Rock%';
SELECT name, artist, genre FROM albums_db.albums WHERE genre = 'Rock';
-- +---------------------------------------+-------------------+-------+
-- | name                                  | artist            | genre |
-- +---------------------------------------+-------------------+-------+
-- | Sgt. Pepper's Lonely Hearts Club Band | The Beatles       | Rock  |
-- | 1                                     | The Beatles       | Rock  |
-- | Abbey Road                            | The Beatles       | Rock  |
-- | Born in the U.S.A.                    | Bruce Springsteen | Rock  |
-- | Supernatural                          | Santana           | Rock  |
-- +---------------------------------------+-------------------+-------+