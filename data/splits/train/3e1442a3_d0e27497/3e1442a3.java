import java.util.*;
public class Main {
	public static void main(String[] args){
		Scanner sc = new Scanner(System.in);

		// long mod = 1_000_000_007L;
		// long mod = 998_244_353L;

		int t = sc.nextInt();

		for ( int zzz=0; zzz<t; zzz++ ) {

			int n = sc.nextInt();

			HashMap<Integer, HashSet<Integer>> adj = new HashMap<>();
			HashMap<HashSet<Integer>, Integer> inv = new HashMap<>();
			boolean f = false;
			for ( int i=0; i<n-1; i++ ) {
				int u = sc.nextInt();
				int v = sc.nextInt();
				if ( adj.containsKey(u) ) {
					HashSet<Integer> s = adj.get(u);
					if ( s.size()>1 ) f = true;
					s.add(v);
					adj.put(u, s);
				} else {
					HashSet<Integer> s = new HashSet<>();
					s.add(v);
					adj.put(u, s);
				}
				if ( adj.containsKey(v) ) {
					HashSet<Integer> s = adj.get(v);
					if ( s.size()>1 ) f = true;
					s.add(u);
					adj.put(v, s);
				} else {
					HashSet<Integer> s = new HashSet<>();
					s.add(u);
					adj.put(v, s);
				}

				HashSet<Integer> si = new HashSet<>();
				si.add(u);
				si.add(v);
				inv.put(si, i);
			}

			if ( f ) {
				System.out.println(-1);
				continue;
			}

			String[] ans = new String[n-1];
			boolean g = false;
			ArrayDeque<Integer> q = new ArrayDeque<>();
			q.addLast(1);
			boolean[] seen = new boolean[n+1];
			while ( q.size()>0 ) {
				int v = q.removeLast();
				seen[v] = true;

				HashSet<Integer> s = adj.get(v);
				for ( int e : s ) {
					if ( seen[e] ) continue;

					HashSet<Integer> st = new HashSet<>();
					st.add(v);
					st.add(e);
					int pos = inv.get(st);
					if ( g ) {
						ans[pos] = "3";
					} else {
						ans[pos] = "2";
					}
					g = !g;
					q.addLast(e);
					break;
				}
			}

			g = true;
			q.addLast(1);
			while ( q.size()>0 ) {
				int v = q.removeLast();
				seen[v] = true;

				HashSet<Integer> s = adj.get(v);
				for ( int e : s ) {
					if ( seen[e] ) continue;

					HashSet<Integer> st = new HashSet<>();
					st.add(v);
					st.add(e);
					int pos = inv.get(st);
					if ( g ) {
						ans[pos] = "3";
					} else {
						ans[pos] = "2";
					}
					g = !g;
					q.addLast(e);
					break;
				}
			}

			System.out.println(String.join(" ", ans));
		}
	}
}
