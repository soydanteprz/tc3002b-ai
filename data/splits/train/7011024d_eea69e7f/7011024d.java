import java.util.*;

public class D {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        ArrayList<Integer> occupied = new ArrayList<>();
        ArrayList<Integer> vacant = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int x = scanner.nextInt();
            if (x == 1)
                occupied.add(i);
            else
                vacant.add(i);
        }

        Solution Solution = new Solution(occupied, vacant);

//        System.out.println(Solution.tabulation());
        System.out.println(Solution.memoization());
    }
}

class Solution {

    ArrayList<Integer> occupied, vacant;
    int x, y;

    public Solution(ArrayList<Integer> occupied, ArrayList<Integer> vacant) {
        this.occupied = occupied;
        this.vacant = vacant;
        x = occupied.size(); y = vacant.size();
    }

    

    

    int memoization() {
        int[][] dp = new int[x][y];
        for (int i = 0; i < x; i++) {
            Arrays.fill(dp[i], -1);
        }
        return memoization(dp, x-1, y-1);
    }

    int memoization(int[][] dp, int n, int m) {
        if(n < 0) {
            return 0;
        }
        if(m < n) {
            return Integer.MAX_VALUE;
        }
        if(dp[n][m] != -1) {
            return dp[n][m];
        }
        int first = memoization(dp, n, m-1);
        int second = memoization(dp, n-1, m-1) + Math.abs(occupied.get(n) - vacant.get(m));
        dp[n][m] = Math.min(first, second);
        return dp[n][m];
    }
}
