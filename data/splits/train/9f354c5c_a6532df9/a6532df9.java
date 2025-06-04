import java.io.*;
import java.util.*;
public class new1{
	
	
	public static void dfs(ArrayList<ArrayList<int[]> > aList1, int p, int u, int[] ans, int v) {
		int[] aa = aList1.get(u).get(0);
		if(aa[0] != p) {
			ans[aa[1]] = v;
			if(v == 2) dfs(aList1, u, aa[0], ans, 3);
			else dfs(aList1, u, aa[0], ans, 2);
		}
		else if(aList1.get(u).size() > 1){
			aa = aList1.get(u).get(1);
			ans[aa[1]] = v;
			if(v == 2) dfs(aList1, u, aa[0], ans, 3);
			else dfs(aList1, u, aa[0], ans, 2);
		}
		
		
	}
	
	public static void main(String[] args)  throws IOException{

		
		BufferedWriter output = new BufferedWriter(new OutputStreamWriter(System.out));
		FastReader s = new FastReader();
		int t = s.nextInt();
		for(int z = 0; z < t; z++) {
			int n = s.nextInt();
			ArrayList<ArrayList<int[]> > aList1 =  new ArrayList<ArrayList<int[]> >(n + 1);
			
	        for (int j = 1; j <= n + 1; j++) { 
	            ArrayList<int[]> list = new ArrayList<>();
	            aList1.add(list);
	        }
		
			for(int j = 0; j < n - 1; j ++) {
				int u = s.nextInt();
				int v = s.nextInt();
				int[] aa = {v, j};
				int[] bb = {u, j};
				aList1.get(u).add(aa);
				aList1.get(v).add(bb);
			}
			int pos = 1; int ind = -1;
			for(int i = 1; i <= n; i++) {
				if(aList1.get(i).size() > 2) {
					pos = 0;
					break;
				}
				if(aList1.get(i).size() == 1) {
					ind = i;
				}
			}
			if(pos == 0) {
				output.write(-1 + "\n");
			}
			else {
				int[] ans = new int[n - 1];
				dfs(aList1, -1, ind, ans, 2);
				for(int i = 0; i < n - 1; i++) {
					output.write(ans[i] + " ");
				}
				output.write("\n");
			}
			
			
			
		}
	   output.flush();
	  }
}


 
	
