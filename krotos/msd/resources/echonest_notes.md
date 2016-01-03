# Utilizing SQLite

First we must generate the database from the provided tab-separated file

```
> ls
train_triplets.txt ...
> sqlite3 train_triplets.db
sqlite> create table plays(user varchar(40), song varchar(18), count unsigned integer);
sqlite> .separator "\t"
sqlite> .import train_triplets.txt plays
```

This takes some time to do, you might want to watch an episode of your favorite anime or what have you.





# Reducing the song space

Next, we create an intermediate table of the 10,000 most **popular** (_not the same as most played_) songs. We do this for speeding up computation of the latent factors, justified by the popularity-bias noted in [Section 3.2 & 5.4 of McFee et al.](http://eceweb.ucsd.edu/~gert/papers/msdc.pdf)

We discriminate _not_ with a sum of play counts, but with the number of unique listeners. Some listeners have ridiculous play counts of single songs that may not reflect the popularity of a song.

This step took around ~16 minutes on an i5-powered laptop.

```
create table popular_songs as
select song, count(user)
from plays
group by song
order by 2 desc
limit 10000;
```

We then can re-generate a subset of the main dump such that each song is constrained to be "popular", which took ~3 minutes.

```
create table popular_plays as
select plays.user, popular_songs.song, plays.count
from plays, popular_songs
where plays.song = popular_songs.song;
```

> Don't feel like processing extra data? You can just get the approximate final results from below by doing this instead. However, this will lead to varying degrees of success. Please continue to read.
> ```
create table subset_plays as
select plays.user, popular_songs.song, plays.count
from plays, popular_songs
where plays.song = popular_songs.song
and (random() % 600) = 0;
```

And from the following,

```
select count(distinct user), count(distinct song) from popular_plays;
```

we see that our subset of data has 1,011,817 distinct users and 10,000 distinct songs. Compared to the original 1,019,318 users and 384,546 songs, _we only see a significant reduction in the number of songs_.

Let's try and reduce the user space!





# Reducing the user space

To make computation easier, we also need to reduce the number of users in our dataset.

We can randomly select user-song-count triplets to drop from our database with a `WHERE` clause utilizing `random()`.

In practice, this method gives varying results as listed below.

> The measurement took under 20 seconds to do.
> ```
select count(distinct user), count(distinct song)
from popular_plays where (random() % modulo) = 0;
```

Subsequent measurements with a modulo of 1,300:

|Distinct users|Distinct songs|
|-------------:|-------------:|
|19,936        |7.352         |
|19,858        |7.359         |
|19,874        |7.371         |

We sample our subset by doing the following. It took me ~30 seconds.

```
create table subset_plays as
select * from popular_plays
where (random() % 1300) = 0;
```
> _"let the samples hit the floor, let the samples hit the floor"_

We can just `drop table subset_plays` and repeat if we are unsatisfied with our ad hoc `count distinct` measurements. Always test after doing work!

In our case, we ended up with 19,965 distinct users and 7,358 distinct songs.

Our latent factor vectors are now computationally less expensive, at the cost of covering a smaller song space (leaving out information of tail-end songs) and providing less trainable data for learning content features in our CNN.
