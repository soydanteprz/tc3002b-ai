import java.util.*;
import java.math.*;
public class Main {
	
	public static class Edge{
		int u;
		int v;
//		int wt;
		Edge(int u, int v){
			this.u = u;
			this.v = v;
//			this.wt = wt;
		}
	}

	public static void main(String[] args) {
		Scanner scn = new Scanner(System.in);
		int t = scn.nextInt();
		StringBuilder sb = new StringBuilder("");
		for(int A=0 ; A<t ; A++) {
			int n = scn.nextInt();
			List<Edge>[] graph = new ArrayList[n];
			for(int i=0 ; i<n ; i++) {
				graph[i] = new ArrayList<>();
			}
			String[] arr = new String[n-1];
			for(int i=0 ; i<n-1 ; i++) {
				int u = scn.nextInt();
				int v = scn.nextInt();
				arr[i] = (u-1) + " " + (v-1);
				graph[u-1].add(new Edge(u-1, v-1));
				graph[v-1].add(new Edge(v-1, u-1));
			}
			
			boolean flag = false;
			int src = 0;
			for(int i=0 ; i<n ; i++) {
				if(graph[i].size() > 2) {
					flag = true;
				}else if(graph[i].size() == 1)
					src = i;
			}
			if(flag) {
				sb.append(-1 + "\n");
			}else {
				Map<String, Integer> hm = new HashMap<>();
				int count = 0;
				int val = 2;
	//			System.out.println(src);
				
				while(count < n) {
					List<Edge> e = graph[src];
					int i=0;
					while(i < e.size() && hm.containsKey(src + " " + e.get(i).v))
						i++;
					if(i < e.size()) {
						int nbr = e.get(i).v;
		//				System.out.println(src + " " + nbr);
						hm.put(src + " " + nbr, val);
						hm.put(nbr + " " + src, val);
						val = 5 - val;
						src = nbr;
					}
						
					count++;
				}
				for(int i=0 ; i<arr.length ; i++) {
					int wt = hm.get(arr[i]);
					sb.append(wt + " ");
				}
				sb.append("\n");
			}
		}
		System.out.println(sb);
	}

}
