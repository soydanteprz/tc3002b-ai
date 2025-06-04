import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringTokenizer;

public class C {

	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws NumberFormatException, IOException {
		BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
		int inputs = Integer.parseInt(in.readLine());
		StringBuilder ans = new StringBuilder();
		while(inputs-->0) {
			int n = Integer.parseInt(in.readLine());
			int[] degree = new int[n];
			boolean bad = false;
			ArrayList<int[]>[] con = new ArrayList[n];
			for(int i = 0; i < n; i++) {
				con[i] = new ArrayList<>();
			}
			for(int i = 0; i < n-1; i++) {
				StringTokenizer st = new StringTokenizer(in.readLine());
				int a = Integer.parseInt(st.nextToken())-1;
				int b = Integer.parseInt(st.nextToken())-1;
				degree[a]++;
				degree[b]++;
				if(degree[a] > 2 || degree[b] > 2)
					 bad = true;
				con[a].add(new int[] {b, i});
				con[b].add(new int[] {a, i});
			}
			if(bad)
				ans.append("-1\n");
			else {
				int[] assign = new int[n-1];
				for(int i = 0; i < degree.length; i++) {
					if(degree[i] == 1) {
						dfs(i, -1, 11, con, assign);
						break;
					}
				}
				for(int i = 0; i < n-2; i++)
					ans.append(assign[i] + " ");
				ans.append(assign[n-2] + "\n");
			}
		}
		System.out.print(ans);

	}
	
	public static void dfs(int curr, int prev, int num, ArrayList<int[]>[] con, int[] a) {
		for(int[] next : con[curr]) {
			if(next[0] != prev) {
				a[next[1]] = 13-num;
				dfs(next[0], curr, 13-num, con, a);
			}
		}
	}

}
