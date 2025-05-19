### Spotify playlist popularity (Jakob)
- Data: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
- Network: we could create an artist-playlist bipartite network
Nodes: two types - playlists and artists
Edges: artist appears in playlist
Node attr: number of followers (playlists), artists?
Goal: analyze success factors in playlist networks and predict number of followers with regression models
Potential research questions: 
Which network metrics most strongly predict playlist popularity
Do different artists/albums on playlist correlate with more followers (diversity)
can and how much of playlist popularity can be explained by the network structure alone

If we manage to train the models and they can succesfully predict the popularity of the playlists (followers), we can turn around the problem so that we can randomly generate a playlist and then predict if it is a good playlist or not.
