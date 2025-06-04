import java.util.*;
import java.math.*;
public class Solution {
	final static int MOD = 1000000007;
	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		int t = Integer.parseInt(in.next());
		for(int tt = 0; tt < t; tt++) {
			int n = Integer.parseInt(in.next());
			int m = Integer.parseInt(in.next());
			char[][] grid = new char[n][m];
			List<Integer> ans = new ArrayList<>();
			for(int i = 0; i < n; i++) {
				grid[i] = in.next().toCharArray();
			}
			int iii = 0;
			int jjj = 0;
			if(n % 2 != 0) {
				iii = 1;
				for(int j = 0; j + 1 < m; j++) {
					if(grid[0][j] == '1') {
						grid[0][j] = '0';
						grid[1][j] = (grid[1][j] == '1') ? '0' : '1';
						grid[1][j + 1] = (grid[1][j + 1] == '1') ? '0' : '1';
						ans.add(1); ans.add(j + 1);
						ans.add(2); ans.add(j + 1);
						ans.add(2); ans.add(j + 2);
					}
					if(grid[0][j + 1] == '1') {
						grid[0][j + 1] = '0';
						grid[1][j] = (grid[1][j] == '1') ? '0' : '1';
						grid[1][j + 1] = (grid[1][j + 1] == '1') ? '0' : '1';
						ans.add(1); ans.add(j + 2);
						ans.add(2); ans.add(j + 1);
						ans.add(2); ans.add(j + 2);
					}
				}
			}
			if(m % 2 != 0) {
				jjj = 1;
				for(int i = (n % 2 == 0) ? 0 : 1; i + 1 < n; i++) {
					if(grid[i][0] == '1') {
						grid[i][0] = '0';
						grid[i][1] = (grid[i][1] == '1') ? '0' : '1';
						grid[i + 1][1] = (grid[i + 1][1] == '1') ? '0' : '1';
						ans.add(i + 1); ans.add(1);
						ans.add(i + 1); ans.add(2);
						ans.add(i + 2); ans.add(2);
					}
					if(grid[i + 1][0] == '1') {
						grid[i + 1][0] = '0';
						grid[i][1] = (grid[i][1] == '1') ? '0' : '1';
						grid[i + 1][1] = (grid[i + 1][1] == '1') ? '0' : '1';
						ans.add(i + 2); ans.add(1);
						ans.add(i + 1); ans.add(2);
						ans.add(i + 2); ans.add(2);
					}
				}
			}
			for(int i = iii; i < n; i += 2) {
				for(int j = jjj; j < m; j += 2) {
					int cnt = count(grid, i, j);
					while(cnt > 0) {
						List<G> g = new ArrayList<>();
						for(int k = i; k < i + 2; k++) {
							for(int l = j; l < j + 2; l++) {
								g.add(new G(grid[k][l], k, l));
							}
						}
						Collections.sort(g);
						int l = 0;
						int r = 0;
						if(cnt == 1 || cnt == 3) {
							l = 1; r = 3;
						}else {
							l = 0; r = 2;
						}
						for(int k = l; k <= r; k++) {
							int ii = g.get(k).i;
							int jj = g.get(k).j;
							grid[ii][jj] = (grid[ii][jj] == '1') ? '0' : '1';
							ans.add(ii + 1);
							ans.add(jj + 1);
						}
						cnt = count(grid, i, j);
					}
				}
			}
			System.out.println(ans.size() / 6);
			for(int i = 0; i < ans.size(); i += 6) {
				for(int j = i; j < i + 6; j++) {
					System.out.print(ans.get(j) + " " );
				}
				System.out.println();
			}
		}
	}
	public static int count(char[][] grid, int i, int j) {
		int cnt = 0;
		for(int k = i; k < i + 2; k++) {
			for(int l = j; l < j + 2; l++) {
				if(grid[k][l] == '1') cnt += 1;
			}
		}
		return cnt;
	}
}
class G implements Comparable <G> {
	Character val;
	int i;
	int j;
	public G(char val, int i, int j) {
		this.val = val;
		this.i = i;
		this.j = j;
	}
	
}
