import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

import static java.lang.Math.*;
import static java.lang.System.out;
import static java.lang.Long.MAX_VALUE;

public final class Main {

    FastReader in;
    StringBuffer sb;
    public static void main(String[] args) {
        new Main().run();
    }

    void run(){
        in= new FastReader();
        start();
    }

    void start(){
        sb= new StringBuffer();
        for(int t=in.nextInt();t>0;t--)
            solve();
        out.print(sb);
    }

    void solve(){
        int n = in.nextInt();
        int[][] mat = new int[n][5];
        for(int i = 0; i<n; i++) {
            char[] pp =in.nextLine().toCharArray();
            for(char c : pp) {
                mat[i][c-'a']++;
            }
            int[] p = new int[5];
            int tt = 0;
            for(int j : mat[i])
                tt +=j;
            for(int j = 0; j<5; j++) {
                mat[i][j]=2*mat[i][j]-tt;
            }
        }
        int ans  = 0;
        for(int i = 0; i<5; i++) {
            int g = check(i,mat, n);
            ans = Math.max(g,ans);
        }
        sb.append(ans).append("\n");
    }

    int check(int i, int[][] mat, int n) {
        ArrayList<Integer> x = new ArrayList<>();
        for(int j = 0; j<n; j++) {
            x.add(mat[j][i]);
        }
        Collections.sort(x);
        int s = 0;
        int cnt = 0;
        int l = n-1;
        while(l>=0) {
            int u = x.get(l);
            if(s+u>0) {
                s+=u;
                cnt++;
            }
            else
                break;
            l--;
        }
        return cnt;
    }

    

    

    void sort(long[] A){
        int n = A.length;
        Random rnd = new Random();
        for(int i=0; i<n; ++i){
            long tmp = A[i];
            int randomPos = i + rnd.nextInt(n-i);
            A[i] = A[randomPos];
            A[randomPos] = tmp;
        }
        Arrays.sort(A);
    }

    

    long MAX= MAX_VALUE;
    int[] precomp= new int[(int) (MAX+1)];
    

    

    

    

    
    

    

    

    
}
