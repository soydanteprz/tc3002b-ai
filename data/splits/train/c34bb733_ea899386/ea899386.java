import java.io.*;
import java.util.*;
 
public class Codeforces {
	public static class Tower implements Comparable<Tower>{
		int val;
		int index;
		public Tower(int ind, int v) {
			val = v;
			index = ind;
		}
		
	}
	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		int cases = Integer.parseInt(br.readLine());
		while(cases-- > 0) {
			String[] str = br.readLine().split(" ");
			int n = Integer.parseInt(str[0]);
			int m = Integer.parseInt(str[1]);
			int x = Integer.parseInt(str[2]);
			int[] h = new int[n];
			str = br.readLine().split(" ");
			for(int i=0; i<n; i++) {
				h[i] = Integer.parseInt(str[i]);
			}
			PriorityQueue<Tower> q = new PriorityQueue<>(m);
			int[] ans = new int[n];
			for(int i=0; i<m; i++) {
				q.add(new Tower(i, h[i]));
				ans[i] = i;
			}
			for(int i=m; i<n; i++) {
				Tower lowest = q.poll();
				lowest.val += h[i];
				ans[i] = lowest.index;
				q.add(lowest);
			}
			System.out.println("YES");
			for(int i=0; i<n; i++) {
				System.out.print((ans[i]+1) + " ");
			}
			System.out.println();
		}
	}
}
