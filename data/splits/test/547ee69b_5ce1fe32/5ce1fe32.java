import java.io.*;
import java.util.*;

public class D{

    static long[][] dp;
    static List<Integer> seats,people;
    static int n1,n2;

    static long solve(int i,int j){
        
        if(i==n1) return 0;
        long ans = Integer.MAX_VALUE;
        if(n1-i>n2-j) return ans;
        if(dp[i][j]!=-1) return dp[i][j];
        
        ans = Math.min(solve(i,j+1),solve(i+1,j+1)+Math.abs(people.get(i)-seats.get(j)));
        // out.printf("%d %d : %ld",i,j,ans);
        return dp[i][j]=ans;
    }

    public static void main(String args[])throws IOException{

        int n=sc.nextInt();
        
        seats = new ArrayList<>(n/2);
        people = new ArrayList<>(n/2);

        for(int i=1;i<=n;i++){
            if(sc.nextInt()==0){
                seats.add(i);
            }else{
                people.add(i);
            }
        }

        n1=people.size();
        n2=seats.size();

        // if(n2==n){
        //     out.println(0);
            
        // }

        dp = new long[n1][n2]; 
        for(int i=0;i<n1;i++)
        Arrays.fill(dp[i],-1);

        long ans =solve(0,0);
        out.println(ans);

        out.close();
    }
    


    


    

    

    
    

    static FastScanner sc = new FastScanner();
    static PrintWriter out =new PrintWriter(System.out);

    
}
