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





# Reducing the dataset

If you don't feel like performing operations on 1,000,000-by-40 dense matrices or 1,000,000-by-1,000,000 diagonal matrices, you can reduce the dataset down to the ~20K users and ~10K songs that [Dieleman et al.](http://papers.nips.cc/paper/5004-deep-content-based-music-recommendation.pdf) does in his work in Section 5.1. Feel free to skip to **We're not done yet!** below to retain the full dataset.

### Reducing the song space

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





# We're not done yet!

> If you are retaining the full dataset, the `subset_plays` references below should just be `plays` table.

Before we get started, now's a good time to `VACCUM` our database up. If you ended up doing a lot of data wrestling, it might benefit you to defragment your database file for the sake of performance. It'll take a while, so load up that second episode of anime (it took me ~15 minutes).

In order to help our collaborative filtering implementation, we need to build a several things:

- a mapping from a matrix index/position to an Echo Nest User ID
- a mapping from a matrix index/position to an Echo Nest Song ID
- storage for computed latent feature vectors

We can build tables to represent each user/song's index as their implicit `ROWID` and store serialized `numpy` latent feature vectors in a `blob` column.

We may also wish to enforce `unique` on the user/song IDs to improve searching performance. The tables will also be used to translate between `numpy` indices and IDs.

```
create table vector_users(user varchar(40) unique, x blob default '');

create table vector_songs(song varchar(18) unique, y blob default '');

insert into vector_users (user, x)
select distinct user, ''
from subset_plays;

insert into vector_songs (song, y)
select distinct song, ''
from subset_plays;
```

> The `x`, `y` notation follows that of [Koren et al.](http://yifanhu.net/PUB/cf.pdf)

While we're at it, we'll add database indexes to `subset_plays`'s user and song IDs as well.

```
create index subset_plays_user_idx on subset_plays (user);
create index subset_plays_song_idx on subset_plays (song);
```
