import java.io.*;
import java.util.*;
public class Ishu
{
    static Scanner scan = new Scanner(System.in);
    static BufferedWriter output = new BufferedWriter(new OutputStreamWriter(System.out)); 
    static void tc() throws Exception
    {
    int n = scan.nextInt();
    String[] str = new String[n];
    int i, j;
    
    for(i=0;i<n;++i)
        str[i] = scan.next();
        
    String ans = "NO";
    for(i=0;i<n;++i)
        {
        String cur = str[i];
        int len = cur.length();
        if(cur.charAt(0) == cur.charAt(len - 1))
            {
            output.write("YES\n");
            output.flush();
            return;
            }
        }
        
    Map<String, Integer> map = new HashMap<String, Integer>();
    Map<String, Integer> par = new HashMap<String, Integer>();
    
    for(i=0;i<n;++i)
       {
       String cur = str[i];
       int len = cur.length();
       if(len == 2)
            {
            if(!map.containsKey(cur))
                map.put(cur, 0);
                
            int value = map.get(cur);
            ++value;
            map.remove(cur);
            map.put(cur, value);
            }
       else 
            {
            String pre = cur.substring(0, 2);
            if(!par.containsKey(pre))
                par.put(pre, 0);
            int value = par.get(pre);
            ++value;
            par.remove(pre);
            par.put(pre, value);
            
            if(!map.containsKey(cur))
                map.put(cur, 0);
            value = map.get(cur);
            ++value;
            map.remove(cur);
            map.put(cur, value);
            }     
       }
       
    for(i=n-1;i>=0;--i)
        {
        String cur = str[i];
        int len = cur.length();
        if(len == 2)
            {
            int value = map.get(cur);
            --value;
            map.remove(cur);
            if(value > 0)
                map.put(cur, value);
            }
        else
            {
            String pre = cur.substring(0, 2);
            int value = par.get(pre);
            --value;
            par.remove(pre);
            if(value > 0)
                par.put(pre, value);
                
            value = map.get(cur);
            --value;
            map.remove(cur);
            if(value > 0)
                map.put(cur, value);
            }    
        
        if(len == 2)
            {
            StringBuffer buffer = new StringBuffer(cur);
            buffer.reverse();
            String rev = String.valueOf(buffer);
            
            if(map.containsKey(rev) || par.containsKey(rev))
                {
                ans = "YES";
                break;
                }    
            }
        else
            {
            String suf = cur.substring(1, 3);
            StringBuffer buffer = new StringBuffer(suf);
            buffer.reverse();
            String rev = String.valueOf(buffer);
            
            if(map.containsKey(rev))
                {
                ans = "YES";
                break;
                }
            
            buffer = new StringBuffer(cur);
            buffer.reverse();
            rev = String.valueOf(buffer);
            
            if(map.containsKey(rev))
                {
                ans = "YES";
                break;
                }
            }
        }
        
    output.write(ans + "\n");
    output.flush();
    }
    
    public static void main(String[] args) throws Exception
    {
    int t = 1;
    t = scan.nextInt();
    while(t-- > 0)
        tc();
    }
}
