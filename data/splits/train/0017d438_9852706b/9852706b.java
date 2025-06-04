import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
public class Main {
    static int modulo=998244353;
    public static void main(String[] args) {
       
        FastScanner in = new FastScanner();
        int test=in.nextInt();
        while(test-->0){
            int n=in.nextInt();
            int m=in.nextInt();
            int x=in.nextInt();
             PriorityQueue<pair> pq=new PriorityQueue<>();
            int arr[]=new int[n];
            for(int i=1;i<=m;i++){
                pq.add(new pair(i,0));
            }
            System.out.println("YES");
            for(int i=0;i<n;i++){
                pair p=pq.poll();
                int a=in.nextInt();
                p.y+=a;
                pq.add(p);
                System.out.print(p.x+" ");
            }
            
            
            
            System.out.println();
        }
        
        
       
        
        
//        int sum=in.nextInt();
//        
//        boolean dp[][]=new boolean[n+1][sum+1];
//        
//        for(int i=0;i<=n;i++){
//            for(int j=0;j<=sum;j++){
//                if(i==0)
//                    dp[i][j]=false;
//                if(j==0)
//                    dp[i][j]=true;
//                    
//            }
//        }
//        
//        for(int i=1;i<=n;i++){
//            for(int j=1;j<=sum;j++){
//                if(arr[i-1]<=j)
//                    dp[i][j]=(dp[i-1][j])||(dp[i-1][j-arr[i-1]]);
//                else
//                    dp[i][j]=dp[i-1][j];
//                
//            }
//        }
//         boolean flag=false;
//        for(int i=0;i<=n;i++){
//            if(dp[i][sum]==true){
//                flag=true;
//                System.out.println(1);
//                break;
//            }
//            
//        }
//        if(!flag)
//            System.out.println(0);
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
}

class pair implements Comparable<pair>
	    {
	        int x;
	        int y;
 
	        pair(int x, int y) {
	            this.x = x;
	            this.y = y;
	        }
	        
}
