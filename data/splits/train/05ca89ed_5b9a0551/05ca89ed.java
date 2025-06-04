import java.io.BufferedReader; 
import java.io.IOException; 
import java.io.InputStreamReader; 
import java.util.Scanner; 
import java.util.StringTokenizer; 
import java.util.*;
  
public class er106c
{ 
    //By shwetank_verma
	public static void main(String[] args) 
    { 
        FastReader sc=new FastReader(); 
        try{
           
                int t=1;
                t=sc.nextInt();
                while(t-->0){
                int n=sc.nextInt();
                long o=n,e=n;
                long maxo=Integer.MAX_VALUE;
                long maxe=Integer.MAX_VALUE;
                long ans=Long.MAX_VALUE;
                long temp=0;
                
                int a[]=new int[n];
                for(int i=0;i<n;i++){
                    a[i]=sc.nextInt();
                }
                for(int i=0;i<n;i++) {
                	if(i%2==1) {
                		temp+=a[i];
                		e--;
                		maxe=Long.min(maxe, a[i]);
                		ans=Long.min(ans,temp+(o*maxo)+(e*maxe));
                	}
                	else {
                		temp+=a[i];
                		o--;
                		maxo=Long.min(maxo, a[i]);
                		ans=Long.min(ans,temp+(o*maxo)+(e*maxe));
                	}
                }
                System.out.println(ans);
                
                
        
                }
           
            
        }catch(Exception e){
            return;
        }
    }
     
    static int mod=1000000007;
    static boolean primes[]=new boolean[1000007];
    static ArrayList<Integer> b=new ArrayList<>();
    
    
    
    
      


  
     
}
