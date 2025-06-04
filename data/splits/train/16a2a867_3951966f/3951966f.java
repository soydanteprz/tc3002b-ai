import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class Armchairs {
    static  ArrayList<Integer> f;
    static ArrayList<Integer> u;
    static int dp[][];
    static int fun(int i, int j){
        if(i == f.size()) return 0;
        if(j == u.size()) return 99999999;
        if(dp[i][j] != -1) return dp[i][j];
        int ans1 = fun(i, j+1);
        int ans2 = fun(i+1, j+1) + Math.abs(f.get(i)-u.get(j));
        return dp[i][j] = Math.min(ans1, ans2);
    }
    private static int solve(int n, int a[]) {
        for (int i = 0; i < n; i++) {
            if (a[i]==0)
                u.add(i);
            else
                f.add(i);
        }

        return fun(0,0);
    }

    public static void main(String[] args)
            throws IOException {
        Scanner s = new Scanner();
        int t = 1;
        StringBuilder ans = new StringBuilder();
        int count = 0;
        while (t-- > 0) {
            int n = s.nextInt();
            int a[] = new int[n];
            dp=new int[n][n];
            for (int i = 0; i < n; i++) {
                a[i]=s.nextInt();
            }
            f=new ArrayList<>();
            u=new ArrayList<>();
            for( int i=0; i<n; i++) Arrays.fill(dp[i],-1);

            ans.append(solve(n, a)).append("\n");
        }
        System.out.println(ans.toString());
    }

    

    

    

    

    

    
}
