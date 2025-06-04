import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;
import java.util.Arrays;
public class Main{  
    public static void main (String[] args){
        FastReader s = new FastReader();
        int t=1;t=s.ni();
        for(int test=1;test<=t;test++){
        int n=s.ni(),k=s.ni();
        int position[]=s.readArray(k),temp[]=s.readArray(k);
        int ans[]=new int[n];
        Arrays.fill(ans,Integer.MAX_VALUE/2);
        for(int i=0;i<k;i++){
            ans[position[i]-1]=temp[i];
        }
        for(int i=1;i<n;i++){
            ans[i]=Math.min(ans[i-1]+1,ans[i]);
        }
        for(int i=n-2;i>=0;i--){
            ans[i]=Math.min(ans[i],ans[i+1]+1);
        }
        for(int i=0;i<n;i++)
        System.out.print(ans[i]+" ");
        System.out.println();
        }
    }
    
    
    
}
